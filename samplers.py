import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import hashlib
import time
from scipy.special import i0e
from scipy.integrate import cumtrapz


class TorsionSampler:
    """
    Base class with Bicubic Differentiable Interpolation and Caching.
    """

    def __init__(self, chain_length, n_chains, cache_dir="cache/ICDF_cache", device="cpu"):
        self._device = device
        self.L = chain_length
        self.N = n_chains
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_or_create_grid(self, weights, k_grid, p_grid):
        # We include 'bicubic' in the hash to ensure cache compatibility
        config_str = f"{self.__class__.__name__}_{weights}_bicubic_dirichlet"
        h = hashlib.md5(config_str.encode()).hexdigest()
        path = os.path.join(self.cache_dir, f"grid_{h}.pkl")

        if os.path.exists(path):
            print(f"\r📦 Loading Crystal {weights} from cache...", end="", flush=True)
            with open(path, 'rb') as f:
                return torch.from_numpy(pickle.load(f)).to(self._device).to(torch.float32)

        print(f"\n⚙️ Precomputing Dirichlet-Anchored Grid for {weights}...")
        grid_np = self.create_2d_grid(weights, k_grid, p_grid)
        with open(path, 'wb') as f:
            pickle.dump(grid_np, f)
        return torch.from_numpy(grid_np).to(self._device).to(torch.float32)

    def _interpolate_2d(self, p_samples, kappas, k_grid, p_grid, icdf_table):
        """
        Differentiable Bicubic Interpolation.
        Returns torsion samples with smooth gradients for diffusion modeling.
        """
        K = kappas.shape[0]
        L, N = self.L, self.N
        table = icdf_table.view(1, 1, *icdf_table.shape)

        # Normalize K to [-1, 1] (y-axis)
        log_k = torch.log10(kappas)
        log_k_grid = torch.log10(k_grid)
        k_min, k_max = log_k_grid[0], log_k_grid[-1]
        k_norm = 2.0 * (log_k - k_min) / (k_max - k_min) - 1.0

        # Normalize P to [-1, 1] (x-axis)
        p_min, p_max = p_grid[0], p_grid[-1]
        p_norm = 2.0 * (p_samples - p_min) / (p_max - p_min) - 1.0

        # Construct Bicubic Sampling Grid
        k_norm_expanded = k_norm.view(1, 1, 1, K).expand(1, L, N, K)
        sampling_grid = torch.stack([p_norm, k_norm_expanded], dim=-1)
        sampling_grid = sampling_grid.view(1, L * N, K, 2).to(torch.float32)

        # Bicubic mode provides quadratic-level smoothness
        samples = F.grid_sample(
            table, sampling_grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True
        )

        return samples.view(1, L, N, K).to(torch.float64)


class VonMisesSampler(TorsionSampler):
    def __init__(self, chain_length, n_chains, tau_c=(0, -120, 120),
                 k_range=(1e-3, 1e7), k_res=1000, p_res=100000, device="cpu"):
        super().__init__(chain_length, n_chains, device=device)
        self.tau_c = np.radians(tau_c)
        self.k_grid = torch.logspace(np.log10(k_range[0]), np.log10(k_range[1]), k_res, device=device)
        self.p_grid = torch.linspace(0, 1, p_res, device=device, dtype=torch.float64)

    def create_2d_grid(self, weights, k_grid, p_grid):
        x = np.linspace(-np.pi, np.pi, 2000)
        k_vals = k_grid.cpu().numpy()
        p_vals = p_grid.cpu().numpy()
        grid_np = np.zeros((len(k_vals), len(p_vals)))

        # 1. Dirichlet Limit: Kappa -> 0 (High Entropy / Uniform)
        grid_np[0, :] = -np.pi + 2 * np.pi * p_vals

        # 2. Dirichlet Limit: Kappa -> Inf (Zero Entropy / Crystal Staircase)
        w_t, w_gp, w_gn = weights
        # Order the staircase thresholds: gn (-120), t (0), gp (+120)
        thresholds = np.cumsum([w_gn, w_t, w_gp])
        staircase = np.zeros_like(p_vals)
        staircase[p_vals < thresholds[0]] = np.radians(-120)
        staircase[(p_vals >= thresholds[0]) & (p_vals < thresholds[1])] = np.radians(0)
        staircase[p_vals >= thresholds[1]] = np.radians(120)
        grid_np[-1, :] = staircase

        # 3. Numerical Integration for Middle Range
        start_time = time.perf_counter()
        for i in range(1, len(k_vals) - 1):
            k = k_vals[i]
            pdf = sum(w * (np.exp(k * (np.cos(x - mu) - 1)) / (2 * np.pi * i0e(k)))
                      for w, mu in zip(weights, self.tau_c))
            cdf = cumtrapz(pdf, x, initial=0)
            cdf /= cdf[-1]
            grid_np[i, :] = np.interp(p_vals, cdf, x)

            if i % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"\r⚙️ Precomputing: {i + 1}/{len(k_vals)} | {elapsed:.2f}s | κ: {k:.2e}", end="")

        print(f"\n✅ Grid Ready.")
        return grid_np

    def __call__(self, weights, kappas):
        icdf_table = self._get_or_create_grid(weights, self.k_grid, self.p_grid)
        kappas = torch.as_tensor(kappas, device=self._device, dtype=torch.float64)
        K = kappas.shape[0]

        # Constant probability seed for smooth 'displacement' tracking
        p_samples = torch.rand((1, self.L, self.N, 1), device=self._device, dtype=torch.float64)
        p_samples = p_samples.expand(-1, -1, -1, K)

        return self._interpolate_2d(p_samples, kappas, self.k_grid, self.p_grid, icdf_table)

    def get_kappa_gradient(self, weights, kappas):
        """
        Computes the partial derivative d(tau) / d(kappa) for each sample point.
        Returns a tensor of shape (1, L, N, K) containing the gradients.
        """
        # 1. Ensure kappas is a tensor and track gradients
        k_tensor = torch.as_tensor(kappas, device=self._device, dtype=torch.float64)
        k_tensor.requires_grad_(True)

        # 2. Forward pass to generate torsions
        # Note: We must use the same probability seeds (p_samples) to get a consistent derivative
        # For a sensitivity analysis, we use the standard __call__ logic.
        icdf_table = self._get_or_create_grid(weights, self.k_grid, self.p_grid)

        # Consistent probability seed (P-space location)
        p_samples = torch.rand((1, self.L, self.N, 1), device=self._device, dtype=torch.float64)
        p_samples = p_samples.expand(-1, -1, -1, k_tensor.shape[0])

        # Perform interpolation
        taus = self._interpolate_2d(p_samples, k_tensor, self.k_grid, self.p_grid, icdf_table)

        # 3. Calculate gradients
        # We want the derivative of each tau_i with respect to its kappa_i.
        # Since grid_sample is independent across the K dimension, we can sum
        # and extract the gradients for all chain elements simultaneously.
        grads = torch.autograd.grad(
            outputs=taus,
            inputs=k_tensor,
            grad_outputs=torch.ones_like(taus),  # Sum of derivatives
            create_graph=True  # Allows for second-order derivatives if needed
        )[0]

        return grads