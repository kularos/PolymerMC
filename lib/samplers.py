"""
Torsion angle samplers for polymer Monte Carlo simulation.

This module provides sampling from Von Mises distributions via
inverse CDF interpolation on precomputed grids. The implementation
separates concerns into:
- Grid computation (pure math)
- Caching logic (I/O)
- Sampling interface (API)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import hashlib
import time
from pathlib import Path
from typing import Tuple
from scipy.special import i0e
from scipy.integrate import cumulative_simpson

from config import SimulationConfig


class GridCache:
    """
    Manages persistent caching of ICDF lookup tables.

    Uses MD5 hashing of configuration parameters to generate unique
    cache keys, ensuring different parameter sets don't collide.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for storing cached grids
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_cache_key(
            self,
            class_name: str,
            weights: Tuple[float, float, float],
            k_range: Tuple[float, float],
            k_res: int,
            p_res: int
    ) -> str:
        """
        Generate unique cache key from configuration.

        Args:
            class_name: Name of sampler class
            weights: Von Mises weight triplet
            k_range: Kappa range (min, max)
            k_res: Kappa grid resolution
            p_res: Probability grid resolution

        Returns:
            MD5 hash string
        """
        config_str = (
            f"{class_name}_"
            f"weights_{weights}_"
            f"k_{k_range}_{k_res}_"
            f"p_{p_res}_"
            f"bicubic_dirichlet"
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def load(self, cache_key: str) -> torch.Tensor | None:
        """
        Load grid from cache if it exists.

        Args:
            cache_key: Unique identifier for this grid

        Returns:
            Cached grid tensor, or None if not found
        """
        cache_path = self.cache_dir / f"grid_{cache_key}.pkl"

        if not cache_path.exists():
            return None

        print(f"📦 Loading cached grid from {cache_path.name}...", end="", flush=True)

        with open(cache_path, 'rb') as f:
            grid_np = pickle.load(f)

        print(" ✓")
        return torch.from_numpy(grid_np).to(torch.float32)

    def save(self, cache_key: str, grid: np.ndarray) -> None:
        """
        Save grid to cache.

        Args:
            cache_key: Unique identifier for this grid
            grid: NumPy array to cache
        """
        cache_path = self.cache_dir / f"grid_{cache_key}.pkl"

        with open(cache_path, 'wb') as f:
            pickle.dump(grid, f)

        print(f"💾 Cached grid to {cache_path.name}")


class VonMisesDistribution:
    """
    Von Mises probability distribution utilities.

    Provides methods for computing PDF, CDF, and inverse CDF (quantile
    function) for mixtures of Von Mises distributions.

    The Von Mises distribution is a circular analog of the normal distribution,
    parameterized by:
    - μ (mu): mean direction (in radians)
    - κ (kappa): concentration parameter (analogous to 1/σ²)

    For κ → 0: uniform distribution (high entropy)
    For κ → ∞: delta function at μ (zero entropy)
    """

    @staticmethod
    def pdf(
            x: np.ndarray,
            mu: float,
            kappa: float
    ) -> np.ndarray:
        """
        Von Mises probability density function.

        PDF = exp(κ cos(x - μ)) / (2π I₀(κ))

        where I₀ is the modified Bessel function of order 0.

        Args:
            x: Evaluation points (radians)
            mu: Mean direction (radians)
            kappa: Concentration parameter

        Returns:
            Probability density at each point
        """
        # Use i0e (exponentially scaled) for numerical stability
        return np.exp(kappa * (np.cos(x - mu) - 1)) / (2 * np.pi * i0e(kappa))

    @staticmethod
    def mixture_pdf(
            x: np.ndarray,
            weights: Tuple[float, float, float],
            mus: Tuple[float, float, float],
            kappa: float
    ) -> np.ndarray:
        """
        PDF of weighted mixture of Von Mises distributions.

        Args:
            x: Evaluation points
            weights: Mixture weights (should sum to 1)
            mus: Mean directions for each component
            kappa: Shared concentration parameter

        Returns:
            Mixed probability density
        """
        pdf = sum(
            w * VonMisesDistribution.pdf(x, mu, kappa)
            for w, mu in zip(weights, mus)
        )
        return pdf

    @staticmethod
    def create_icdf_grid(
            weights: Tuple[float, float, float],
            mus: Tuple[float, float, float],
            k_grid: np.ndarray,
            p_grid: np.ndarray,
            verbose: bool = True
    ) -> np.ndarray:
        """
        Precompute inverse CDF (quantile function) on 2D grid.

        Creates a lookup table mapping (probability, kappa) → torsion angle
        for efficient sampling via interpolation.

        Args:
            weights: Mixture weights for Von Mises components
            mus: Mean directions for each component (radians)
            k_grid: Kappa values (concentration parameters)
            p_grid: Probability values in [0, 1]
            verbose: Print progress updates

        Returns:
            Grid of shape (len(k_grid), len(p_grid)) containing
            torsion angles in radians

        Notes:
            - Uses Dirichlet boundary conditions at kappa extremes
            - At κ=0: uniform distribution (ICDF = -π + 2π*p)
            - At κ=∞: discrete mixture (staircase function)
            - Middle values: numerical integration of CDF
        """
        x = np.linspace(-np.pi, np.pi, 2000)
        grid = np.zeros((len(k_grid), len(p_grid)))

        # Boundary 1: κ → 0 (uniform distribution)
        grid[0, :] = -np.pi + 2 * np.pi * p_grid

        # Boundary 2: κ → ∞ (discrete mixture - staircase)
        w_t, w_gp, w_gn = weights
        thresholds = np.cumsum([w_gn, w_t, w_gp])

        staircase = np.zeros_like(p_grid)
        staircase[p_grid < thresholds[0]] = mus[2]  # Gauche-
        staircase[(p_grid >= thresholds[0]) & (p_grid < thresholds[1])] = mus[0]  # Trans
        staircase[p_grid >= thresholds[1]] = mus[1]  # Gauche+
        grid[-1, :] = staircase

        # Interior: numerical integration
        if verbose:
            print("⚙️  Precomputing ICDF grid via numerical integration...")
            start_time = time.perf_counter()

        for i in range(1, len(k_grid) - 1):
            kappa = k_grid[i]

            # Compute PDF and integrate to get CDF
            pdf = VonMisesDistribution.mixture_pdf(x, weights, mus, kappa)
            cdf = cumulative_simpson(pdf, x=x, initial=0)
            cdf /= cdf[-1]  # Normalize

            # Invert CDF via interpolation
            grid[i, :] = np.interp(p_grid, cdf, x)

            # Progress updates
            if verbose and i % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"\r  Progress: {i}/{len(k_grid) - 2} | "
                    f"Time: {elapsed:.2f}s | κ: {kappa:.2e}",
                    end=""
                )

        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"\n✅ Grid complete in {elapsed:.2f}s")

        return grid


class VonMisesSampler:
    """
    Efficient sampler for Von Mises mixture distributions.

    Uses bicubic interpolation on precomputed ICDF grids to generate
    torsion angle samples. Supports batched sampling across multiple
    concentration parameters simultaneously.

    The sampling process:
    1. Generate uniform random probabilities
    2. Look up corresponding torsion angles via 2D interpolation
       on the (probability, kappa) grid
    3. Return differentiable samples (for gradient-based methods)
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize Von Mises sampler.

        Args:
            config: Simulation configuration containing:
                - chain_length, n_chains: Sample dimensions
                - k_range, k_res, p_res: Grid parameters
                - tau_centers: Mean directions (in degrees)
                - cache_dir: Directory for grid caching
        """
        self.config = config
        self.device = config.torch_device
        self.L = config.chain_length
        self.N = config.n_chains

        # Convert tau centers from degrees to radians
        self.tau_c = config.tau_centers_radians

        # Create kappa grid (logarithmic spacing)
        self.k_grid = torch.logspace(
            np.log10(config.k_range[0]),
            np.log10(config.k_range[1]),
            config.k_res,
            device=self.device
        )

        # Create probability grid (linear spacing)
        self.p_grid = torch.linspace(
            0, 1,
            config.p_res,
            device=self.device,
            dtype=torch.float64
        )

        # Initialize cache manager
        self.cache = GridCache(config.cache_dir)

    def _get_or_create_grid(
            self,
            weights: Tuple[float, float, float]
    ) -> torch.Tensor:
        """
        Load grid from cache or compute if needed.

        Args:
            weights: Von Mises mixture weights

        Returns:
            ICDF grid tensor on device
        """
        # Generate cache key
        cache_key = self.cache._make_cache_key(
            class_name=self.__class__.__name__,
            weights=weights,
            k_range=self.config.k_range,
            k_res=self.config.k_res,
            p_res=self.config.p_res
        )

        # Try loading from cache
        grid = self.cache.load(cache_key)

        if grid is not None:
            return grid.to(self.device)

        # Compute new grid
        print(f"⚙️  Computing new ICDF grid for weights={weights}")
        grid_np = VonMisesDistribution.create_icdf_grid(
            weights=weights,
            mus=self.tau_c,
            k_grid=self.k_grid.cpu().numpy(),
            p_grid=self.p_grid.cpu().numpy(),
            verbose=True
        )

        # Cache for future use
        self.cache.save(cache_key, grid_np)

        return torch.from_numpy(grid_np).to(self.device).to(torch.float32)

    def _interpolate_2d(
            self,
            p_samples: torch.Tensor,
            kappas: torch.Tensor,
            icdf_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform bicubic interpolation on ICDF grid.

        Maps (probability, kappa) pairs to torsion angles using
        differentiable bicubic interpolation for smooth gradients.

        Args:
            p_samples: Probability samples, shape (1, L, N, K)
            kappas: Concentration parameters, shape (K,)
            icdf_grid: Precomputed ICDF table

        Returns:
            Torsion angle samples, shape (1, L, N, K)
        """
        K = kappas.shape[0]

        # Reshape grid for grid_sample: (1, 1, k_res, p_res)
        grid_4d = icdf_grid.view(1, 1, *icdf_grid.shape)

        # Normalize kappa to [-1, 1] (y-axis in grid_sample)
        log_k = torch.log10(kappas)
        log_k_grid = torch.log10(self.k_grid)
        k_min, k_max = log_k_grid[0], log_k_grid[-1]
        k_norm = 2.0 * (log_k - k_min) / (k_max - k_min) - 1.0

        # Normalize probability to [-1, 1] (x-axis in grid_sample)
        p_min, p_max = self.p_grid[0], self.p_grid[-1]
        p_norm = 2.0 * (p_samples - p_min) / (p_max - p_min) - 1.0

        # Construct sampling grid: (1, L*N, K, 2)
        # Last dimension is [x, y] = [probability, kappa]
        k_norm_expanded = k_norm.view(1, 1, 1, K).expand(1, self.L, self.N, K)
        sampling_grid = torch.stack([p_norm, k_norm_expanded], dim=-1)
        sampling_grid = sampling_grid.view(1, self.L * self.N, K, 2).to(torch.float32)

        # Bicubic interpolation (smooth, differentiable)
        samples = F.grid_sample(
            grid_4d,
            sampling_grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True
        )

        # Reshape back to (1, L, N, K)
        return samples.view(1, self.L, self.N, K).to(torch.float64)

    def __call__(
            self,
            weights: Tuple[float, float, float],
            kappas: list[float] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Sample torsion angles from Von Mises mixture.

        Args:
            weights: Mixture weights (T, Gp, Gn) summing to 1
            kappas: Concentration parameters, length K

        Returns:
            Torsion angle samples, shape (1, L, N, K)

        Example:
            >>> sampler = VonMisesSampler(config)
            >>> weights = (0.6, 0.2, 0.2)  # 60% trans, 20% each gauche
            >>> kappas = [1e-2, 1e0, 1e2, 1e4]
            >>> samples = sampler(weights, kappas)
            >>> samples.shape
            torch.Size([1, 128, 8, 4])
        """
        # Load or compute ICDF grid
        icdf_grid = self._get_or_create_grid(weights)

        # Convert kappas to tensor
        kappas = torch.as_tensor(
            kappas,
            device=self.device,
            dtype=torch.float64
        )
        K = kappas.shape[0]

        # Generate uniform random probabilities
        # Use same probability for all kappa values to enable tracking
        p_samples = torch.rand(
            (1, self.L, self.N, 1),
            device=self.device,
            dtype=torch.float64
        )
        p_samples = p_samples.expand(-1, -1, -1, K)

        # Interpolate to get torsion angles
        return self._interpolate_2d(p_samples, kappas, icdf_grid)

    def get_kappa_gradient(
            self,
            weights: Tuple[float, float, float],
            kappas: list[float] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient of torsion angles with respect to kappa.

        Useful for sensitivity analysis and gradient-based optimization.

        Args:
            weights: Mixture weights
            kappas: Concentration parameters

        Returns:
            Gradients ∂τ/∂κ, shape (1, L, N, K)
        """
        # Ensure gradients are tracked
        k_tensor = torch.as_tensor(
            kappas,
            device=self.device,
            dtype=torch.float64
        )
        k_tensor.requires_grad_(True)

        # Forward pass
        icdf_grid = self._get_or_create_grid(weights)

        p_samples = torch.rand(
            (1, self.L, self.N, 1),
            device=self.device,
            dtype=torch.float64
        )
        p_samples = p_samples.expand(-1, -1, -1, k_tensor.shape[0])

        taus = self._interpolate_2d(p_samples, k_tensor, icdf_grid)

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=taus,
            inputs=k_tensor,
            grad_outputs=torch.ones_like(taus),
            create_graph=True
        )[0]

        return grads
