"""
Torsion angle samplers for polymer Monte Carlo simulation.

This module provides sampling from Von Mises distributions via
inverse CDF interpolation on precomputed grids.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple
from scipy.special import i0e
from scipy.integrate import cumulative_simpson

from .config import SimulationConfig
from .core import GridCache


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
                    f"\r  Progress: {i}/{len(k_grid)-2} | "
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
                - pL: Total chain length parameter (generates 2^pL monomers)
                - pS_range, pS_res, p_res: Grid parameters (entropy parameterization)
                - tau_centers: Mean directions (in degrees)
                - cache_dir: Directory for grid caching
        """
        self.config = config
        self.device = config.torch_device
        self.L = config.total_length  # Full chain: 2^pL monomers

        # Convert tau centers from degrees to radians
        self.tau_c = config.tau_centers_radians

        # Create pS grid (entropy parameterization)
        self.pS_grid = torch.linspace(
            config.pS_range[0],
            config.pS_range[1],
            config.pS_res,
            device=self.device,
            dtype=torch.float64
        )

        # Convert to kappa grid for numerical calculations
        # κ = 10^(7 - pS), so higher pS = lower kappa = higher entropy
        self.kappa_grid = 10 ** (7 - self.pS_grid)

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
        # Try loading from cache with pS parameters
        grid = self.cache.load(
            class_name=self.__class__.__name__,
            weights=weights,
            pS_range=self.config.pS_range,
            pS_res=self.config.pS_res,
            p_res=self.config.p_res
        )

        if grid is not None:
            return grid.to(self.device)

        # Compute new grid using kappa values
        print(f"⚙️  Computing new ICDF grid for weights={weights}")
        grid_np = VonMisesDistribution.create_icdf_grid(
            weights=weights,
            mus=self.tau_c,
            k_grid=self.kappa_grid.cpu().numpy(),
            p_grid=self.p_grid.cpu().numpy(),
            verbose=True
        )

        # Save to cache with pS parameters
        self.cache.save(
            grid_np,
            class_name=self.__class__.__name__,
            weights=weights,
            pS_range=self.config.pS_range,
            pS_res=self.config.pS_res,
            p_res=self.config.p_res
        )

        return torch.from_numpy(grid_np).to(self.device).to(torch.float32)

    def _interpolate_2d(
        self,
        p_samples: torch.Tensor,
        pS_values: torch.Tensor,
        icdf_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform bicubic interpolation on ICDF grid.

        Maps (probability, pS) pairs to torsion angles using
        differentiable bicubic interpolation for smooth gradients.

        Args:
            p_samples: Probability samples, shape (1, L, K)
            pS_values: Entropy parameters pS, shape (K,)
            icdf_grid: Precomputed ICDF table

        Returns:
            Torsion angle samples, shape (1, L, K)
        """
        K = pS_values.shape[0]

        # Reshape grid for grid_sample: (1, 1, pS_res, p_res)
        grid_4d = icdf_grid.view(1, 1, *icdf_grid.shape)

        # Normalize pS to [-1, 1] (y-axis in grid_sample)
        pS_min, pS_max = self.pS_grid[0], self.pS_grid[-1]
        pS_norm = 2.0 * (pS_values - pS_min) / (pS_max - pS_min) - 1.0

        # Normalize probability to [-1, 1] (x-axis in grid_sample)
        p_min, p_max = self.p_grid[0], self.p_grid[-1]
        p_norm = 2.0 * (p_samples - p_min) / (p_max - p_min) - 1.0

        # Construct sampling grid: (1, L, K, 2)
        # Last dimension is [x, y] = [probability, pS]
        pS_norm_expanded = pS_norm.view(1, 1, K).expand(1, self.L, K)
        sampling_grid = torch.stack([p_norm, pS_norm_expanded], dim=-1)
        sampling_grid = sampling_grid.view(1, self.L, K, 2).to(torch.float32)

        # Bicubic interpolation (smooth, differentiable)
        samples = F.grid_sample(
            grid_4d,
            sampling_grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True
        )

        # Reshape back to (1, L, K)
        return samples.view(1, self.L, K).to(torch.float64)

    def __call__(
        self,
        weights: Tuple[float, float, float],
        pS_values: list[float] | np.ndarray | torch.Tensor = None,
        kappas: list[float] | np.ndarray | torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample torsion angles from Von Mises mixture.

        Generates a single long chain of length 2^pL that can be
        subdivided later via pGamma for hierarchical analysis.

        Args:
            weights: Mixture weights (T, Gp, Gn) summing to 1
            pS_values: Entropy parameters (preferred), length K
                      Higher pS = higher entropy = more flexible
                      pS = 7 - log10(κ)
            kappas: Concentration parameters (legacy), length K
                   If provided, will be converted to pS internally

        Returns:
            Torsion angle samples, shape (1, 2^pL, K)
            Single chain that can be bisected for analysis

        Example:
            >>> config = SimulationConfig(pL=10)  # 1024 monomers
            >>> sampler = VonMisesSampler(config)
            >>> weights = (0.6, 0.2, 0.2)

            >>> # Sample one long chain
            >>> pS_vals = [5.0, 7.0, 9.0]
            >>> samples = sampler(weights, pS_values=pS_vals)
            >>> samples.shape
            torch.Size([1, 1024, 3])  # One 1024-monomer chain, 3 pS values
        """
        # Handle input: prefer pS_values, fall back to kappas
        if pS_values is None and kappas is None:
            raise ValueError("Must provide either pS_values or kappas")

        if pS_values is not None and kappas is not None:
            raise ValueError("Provide either pS_values OR kappas, not both")

        # Convert kappas to pS if needed (backward compatibility)
        if kappas is not None:
            kappas_array = np.asarray(kappas)
            pS_values = self.config.kappa_to_ps(kappas_array)

        # Load or compute ICDF grid
        icdf_grid = self._get_or_create_grid(weights)

        # Convert pS to tensor
        pS_tensor = torch.as_tensor(
            pS_values,
            device=self.device,
            dtype=torch.float64
        )
        K = pS_tensor.shape[0]

        # Generate uniform random probabilities for ONE chain
        # Shape: (1, L, 1) then expand to (1, L, K)
        p_samples = torch.rand(
            (1, self.L, 1),
            device=self.device,
            dtype=torch.float64
        )
        p_samples = p_samples.expand(-1, -1, K)

        # Interpolate to get torsion angles
        return self._interpolate_2d(p_samples, pS_tensor, icdf_grid)

    def get_pS_gradient(
        self,
        weights: Tuple[float, float, float],
        pS_values: list[float] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient of torsion angles with respect to pS.

        Useful for sensitivity analysis: shows how chain structure
        changes with entropy parameter.

        Args:
            weights: Mixture weights
            pS_values: Entropy parameters

        Returns:
            Gradients ∂τ/∂pS, shape (1, L, K)
        """
        # Ensure gradients are tracked
        pS_tensor = torch.as_tensor(
            pS_values,
            device=self.device,
            dtype=torch.float64
        )
        pS_tensor.requires_grad_(True)

        # Forward pass
        icdf_grid = self._get_or_create_grid(weights)

        p_samples = torch.rand(
            (1, self.L, 1),
            device=self.device,
            dtype=torch.float64
        )
        p_samples = p_samples.expand(-1, -1, pS_tensor.shape[0])

        taus = self._interpolate_2d(p_samples, pS_tensor, icdf_grid)

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=taus,
            inputs=pS_tensor,
            grad_outputs=torch.ones_like(taus),
            create_graph=True
        )[0]

        return grads

    def get_kappa_gradient(
        self,
        weights: Tuple[float, float, float],
        kappas: list[float] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient of torsion angles with respect to kappa.

        Legacy method - prefer get_pS_gradient for clearer interpretation.

        Args:
            weights: Mixture weights
            kappas: Concentration parameters

        Returns:
            Gradients ∂τ/∂κ, shape (1, L, K)
        """
        # Convert kappa to pS
        kappas_array = np.asarray(kappas)
        pS_values = self.config.kappa_to_ps(kappas_array)

        # Get pS gradient
        dτ_dpS = self.get_pS_gradient(weights, pS_values)

        # Apply chain rule: dτ/dκ = (dτ/dpS) * (dpS/dκ)
        # pS = 7 - log10(κ), so dpS/dκ = -1/(κ ln(10))
        kappas_tensor = torch.as_tensor(
            kappas_array,
            device=self.device,
            dtype=torch.float64
        )
        dpS_dκ = -1 / (kappas_tensor * np.log(10))

        # Broadcast and multiply
        dpS_dκ = dpS_dκ.view(1, 1, -1)
        dτ_dκ = dτ_dpS * dpS_dκ

        return dτ_dκ