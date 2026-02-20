"""
Torsion angle samplers for polymer Monte Carlo simulation.

Provides two sampler families on the manifold hierarchy:
  - VonMisesSampler : S¹ (circle)  — scalar torsion angles
  - FisherSampler   : S² (sphere)  — 3D unit bond vectors

VonMisesSampler is the degenerate limit of FisherSampler when
κ_θ → ∞ (bond angle frozen at tetrahedral value).

Class hierarchy:
    BaseSampler (abstract)
    ├── VonMisesSampler   samples τ ∈ S¹, output (1, 2^pL, K)
    └── FisherSampler     samples b̂ ∈ S², output (3, 2^pL, K)
                          called per-step with local frame state

Distribution utilities:
    VonMisesDistribution  — PDF, mixture PDF, ICDF grid construction
    FisherDistribution    — PDF, closed-form polar sampler,
                            azimuthal sampler (reuses VonMises on S¹)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from scipy.special import i0e
from scipy.integrate import cumulative_simpson

from .config import SimulationConfig
from .core import GridCache, rodrigues_rotation


# =============================================================================
# Von Mises distribution utilities (S¹)
# =============================================================================

class VonMisesDistribution:
    """
    Von Mises probability distribution utilities on S¹.

    Provides PDF, mixture PDF, and ICDF grid construction for
    mixtures of Von Mises distributions on the circle.

    Parameterization:
        μ (mu):   mean direction in radians
        κ (kappa): concentration  (κ→0: uniform, κ→∞: delta at μ)
    """

    @staticmethod
    def pdf(
        x: np.ndarray,
        mu: float,
        kappa: float
    ) -> np.ndarray:
        """
        Von Mises PDF: exp(κ cos(x-μ)) / (2π I₀(κ))

        Args:
            x:     Evaluation points (radians)
            mu:    Mean direction (radians)
            kappa: Concentration parameter

        Returns:
            Probability density at each point
        """
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
            x:       Evaluation points
            weights: Mixture weights (should sum to 1)
            mus:     Mean directions for each component (radians)
            kappa:   Shared concentration parameter

        Returns:
            Mixed probability density
        """
        return sum(
            w * VonMisesDistribution.pdf(x, mu, kappa)
            for w, mu in zip(weights, mus)
        )

    @staticmethod
    def create_icdf_grid(
        weights: Tuple[float, float, float],
        mus: Tuple[float, float, float],
        k_grid: np.ndarray,
        p_grid: np.ndarray,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Precompute inverse CDF on a (kappa × probability) grid.

        Args:
            weights: Mixture weights for Von Mises components
            mus:     Mean directions (radians)
            k_grid:  Kappa values (concentration parameters)
            p_grid:  Probability values in [0, 1]
            verbose: Print progress updates

        Returns:
            Grid of shape (len(k_grid), len(p_grid)) — torsion angles (radians)

        Boundary conditions:
            κ → 0:  uniform  → ICDF = -π + 2π·p
            κ → ∞:  discrete → staircase at conformer angles
        """
        x = np.linspace(-np.pi, np.pi, 2000)
        grid = np.zeros((len(k_grid), len(p_grid)))

        # Boundary 1: κ → 0 (uniform on circle)
        grid[0, :] = -np.pi + 2 * np.pi * p_grid

        # Boundary 2: κ → ∞ (discrete mixture — staircase)
        w_t, w_gp, w_gn = weights
        thresholds = np.cumsum([w_gn, w_t, w_gp])
        staircase = np.zeros_like(p_grid)
        staircase[p_grid < thresholds[0]] = mus[2]                                      # Gauche-
        staircase[(p_grid >= thresholds[0]) & (p_grid < thresholds[1])] = mus[0]        # Trans
        staircase[p_grid >= thresholds[1]] = mus[1]                                     # Gauche+
        grid[-1, :] = staircase

        # Interior: numerical CDF integration
        if verbose:
            print("⚙️  Precomputing Von Mises ICDF grid...")
            start_time = time.perf_counter()

        for i in range(1, len(k_grid) - 1):
            kappa = k_grid[i]
            pdf = VonMisesDistribution.mixture_pdf(x, weights, mus, kappa)
            cdf = cumulative_simpson(pdf, x=x, initial=0)
            cdf /= cdf[-1]
            grid[i, :] = np.interp(p_grid, cdf, x)

            if verbose and i % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"\r  Progress: {i}/{len(k_grid)-2} | "
                    f"Time: {elapsed:.2f}s | κ: {kappa:.2e}",
                    end=""
                )

        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"\n✅ Von Mises ICDF grid complete in {elapsed:.2f}s")

        return grid


# =============================================================================
# Fisher distribution utilities (S²)
# =============================================================================

class FisherDistribution:
    """
    Von Mises–Fisher distribution utilities on S².

    The vMF distribution on the 2-sphere is parameterized by:
        μ̂ (mu):   mean direction — unit vector in ℝ³
        κ (kappa): concentration  (κ→0: uniform on S², κ→∞: delta at μ̂)

    PDF:  f(x̂; μ̂, κ) = κ / (4π sinh κ) · exp(κ μ̂·x̂)

    For a mixture of Fishers with azimuthally-placed means, the
    marginal distributions factorize as:
        Polar angle θ:     Fisher polar marginal (closed-form ICDF)
        Azimuthal angle φ: Von Mises mixture on S¹  (reuses VonMisesDistribution)

    This factorization is exact for a single component and approximate
    for mixtures; it becomes exact in the limits κ_θ → 0 and κ_θ → ∞.
    For intermediate κ_θ the means of adjacent components are well-separated
    and the approximation is very accurate.

    Validation limit:
        κ_θ → ∞  →  bond angle frozen at tetrahedral value
                 →  Fisher on S² degenerates to Von Mises on S¹
    """

    @staticmethod
    def pdf(
        cos_angle: np.ndarray,
        kappa: float
    ) -> np.ndarray:
        """
        Von Mises–Fisher PDF as a function of cos(angle from mean).

        f(x̂) = κ / (4π sinh κ) · exp(κ cos θ)

        Args:
            cos_angle: cos(θ) where θ = angle from mean direction
            kappa:     Concentration parameter

        Returns:
            Probability density values
        """
        if kappa < 1e-10:
            # Uniform on S²
            return np.ones_like(cos_angle) / (4 * np.pi)
        return kappa / (4 * np.pi * np.sinh(kappa)) * np.exp(kappa * cos_angle)

    @staticmethod
    def sample_polar(
        kappa_theta: float | np.ndarray,
        n: int,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Sample polar angle θ from Fisher marginal using closed-form ICDF.

        The marginal distribution of cos θ for a Fisher distribution
        has the closed-form inverse CDF (Wood 1994):

            cos θ = 1 + log(u + (1-u)·exp(-2κ)) / κ

        where u ~ Uniform[0, 1].

        At κ_θ → 0:  θ ~ Uniform on [0, π]  (uniform on S²)
        At κ_θ → ∞:  θ → 0                  (all bonds along mean)

        Args:
            kappa_theta: Polar concentration parameter(s), scalar or shape (K,)
            n:           Number of samples to draw
            rng:         Optional NumPy random generator for reproducibility

        Returns:
            Polar angles in [0, π], shape (n,) if kappa_theta is scalar
            or (n, K) if kappa_theta is an array of length K
        """
        if rng is None:
            rng = np.random.default_rng()

        kappa_theta = np.asarray(kappa_theta)
        scalar_input = kappa_theta.ndim == 0
        kappa_theta = np.atleast_1d(kappa_theta)  # (K,)

        # u ~ Uniform[0, 1], shape (n, K)
        u = rng.uniform(0.0, 1.0, size=(n, len(kappa_theta)))

        # Closed-form ICDF for cos θ
        # Numerically stable: separate large-κ and small-κ cases
        cos_theta = np.where(
            kappa_theta[np.newaxis, :] > 1e-6,
            1.0 + np.log(
                u + (1.0 - u) * np.exp(
                    np.clip(-2.0 * kappa_theta[np.newaxis, :], -500, 0)
                )
            ) / kappa_theta[np.newaxis, :],
            # κ → 0: cos θ ~ Uniform[-1, 1]
            2.0 * u - 1.0
        )

        # Clamp to [-1, 1] for numerical safety before arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # (n, K)

        return theta[:, 0] if scalar_input else theta

    @staticmethod
    def sample_azimuthal(
        weights: Tuple[float, float, float],
        mus: Tuple[float, float, float],
        kappa_phi: float | np.ndarray,
        n: int,
        p_grid: np.ndarray,
        icdf_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Sample azimuthal angle φ from Von Mises mixture on S¹.

        The azimuthal marginal of a Fisher mixture with components
        placed at Trans/Gauche+/Gauche- azimuths is a Von Mises mixture
        on the azimuthal circle — identical to the VonMisesSampler case.

        Reuses the precomputed ICDF grid from VonMisesDistribution.

        Args:
            weights:   Mixture weights (T, Gp, Gn)
            mus:       Conformer azimuths in radians (Trans, Gauche+, Gauche-)
            kappa_phi: Azimuthal concentration, scalar or shape (K,)
            n:         Number of samples
            p_grid:    Probability grid used to build icdf_grid, shape (p_res,)
            icdf_grid: Precomputed ICDF table, shape (pS_res, p_res)

        Returns:
            Azimuthal angles in [-π, π], shape (n,) or (n, K)
        """
        kappa_phi = np.asarray(kappa_phi)
        scalar_input = kappa_phi.ndim == 0
        kappa_phi = np.atleast_1d(kappa_phi)  # (K,)
        K = len(kappa_phi)

        # Draw uniform probabilities: (n, K)
        p = np.random.uniform(0.0, 1.0, size=(n, K))

        # Interpolate ICDF grid at each kappa_phi value
        # icdf_grid rows are indexed by kappa — find nearest row for each K
        # (For the FisherSampler we use direct numpy interpolation since
        #  this is called per-step, not in a big batched tensor operation)
        phi = np.zeros((n, K))
        for k_idx in range(K):
            phi[:, k_idx] = np.interp(p[:, k_idx], p_grid, icdf_grid[k_idx])

        return phi[:, 0] if scalar_input else phi

    @staticmethod
    def build_local_frame(
        curr_bond: torch.Tensor,
        prev_bond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct orthonormal local frame (e_z, e_x, e_y) at each step.

        The local frame is defined by:
            e_z = curr_bond (normalized) — polar axis, "forward"
            e_x = component of prev_bond perpendicular to e_z — torsion reference
            e_y = e_z × e_x — completes right-handed frame

        This frame lets us express a Fisher sample in spherical coords
        (θ from e_z, φ from e_x in the e_x/e_y plane) and rotate it
        into the world frame.

        Args:
            curr_bond: Current bond direction, shape (3, K)
            prev_bond: Previous bond direction, shape (3, K)

        Returns:
            e_z: shape (3, K) — polar axis (= curr_bond normalized)
            e_x: shape (3, K) — azimuthal reference (torsion zero)
            e_y: shape (3, K) — completes frame
        """
        # e_z: normalize current bond
        e_z = curr_bond / torch.linalg.norm(curr_bond, dim=0, keepdim=True)

        # e_x: prev_bond projected perpendicular to e_z
        dot = (prev_bond * e_z).sum(dim=0, keepdim=True)   # (1, K)
        e_x_raw = prev_bond - dot * e_z                     # (3, K)
        norm_x = torch.linalg.norm(e_x_raw, dim=0, keepdim=True)

        # Guard: if prev_bond is parallel to curr_bond (degenerate at init),
        # pick an arbitrary perpendicular vector
        degenerate = (norm_x < 1e-8).squeeze(0)  # (K,)
        if degenerate.any():
            # Build a perpendicular via cross with a non-parallel axis
            fallback = torch.zeros_like(e_z)
            fallback[0] = 1.0  # x-axis
            # For degenerate cases, use cross(e_z, x-axis) or cross(e_z, y-axis)
            fb_cross = torch.linalg.cross(e_z, fallback, dim=0)
            fb_norm = torch.linalg.norm(fb_cross, dim=0, keepdim=True).clamp(min=1e-8)
            fb_cross = fb_cross / fb_norm
            # Where still degenerate (e_z parallel to x), try y-axis
            fallback2 = torch.zeros_like(e_z)
            fallback2[1] = 1.0
            fb_cross2 = torch.linalg.cross(e_z, fallback2, dim=0)
            fb_norm2 = torch.linalg.norm(fb_cross2, dim=0, keepdim=True).clamp(min=1e-8)
            fb_cross2 = fb_cross2 / fb_norm2
            # Pick whichever cross product has larger norm
            use_fb2 = (torch.linalg.norm(fb_cross, dim=0) <
                       torch.linalg.norm(fb_cross2, dim=0))
            fallback_vec = torch.where(use_fb2.unsqueeze(0), fb_cross2, fb_cross)
            e_x_raw = torch.where(degenerate.unsqueeze(0), fallback_vec, e_x_raw)
            norm_x = torch.linalg.norm(e_x_raw, dim=0, keepdim=True).clamp(min=1e-8)

        e_x = e_x_raw / norm_x

        # e_y: complete right-handed frame
        e_y = torch.linalg.cross(e_z, e_x, dim=0)
        e_y = e_y / torch.linalg.norm(e_y, dim=0, keepdim=True)

        return e_z, e_x, e_y

    @staticmethod
    def spherical_to_world(
        theta: torch.Tensor,
        phi: torch.Tensor,
        e_z: torch.Tensor,
        e_x: torch.Tensor,
        e_y: torch.Tensor,
        bond_angle: float,
    ) -> torch.Tensor:
        """
        Convert (θ, φ) in local frame to unit bond vector in world frame.

        The tetrahedral bond angle α is incorporated by offsetting θ:
            θ_total = α + θ_fisher
        where α = arccos(alignment) sets the mean polar angle and
        θ_fisher is the Fisher polar deviation around that mean.

        In local spherical coordinates:
            x_local = sin(θ_total) cos(φ)  ·  e_x
                    + sin(θ_total) sin(φ)  ·  e_y
                    + cos(θ_total)         ·  e_z

        Args:
            theta:      Fisher polar deviation, shape (K,)
            phi:        Azimuthal angle, shape (K,)
            e_z:        Polar axis (curr_bond), shape (3, K)
            e_x:        Torsion reference, shape (3, K)
            e_y:        Frame completion, shape (3, K)
            bond_angle: Mean polar offset α = arccos(alignment)

        Returns:
            Unit bond vectors in world frame, shape (3, K)
        """
        # Total polar angle = tetrahedral offset + Fisher deviation
        theta_total = bond_angle + theta   # (K,)

        sin_t = torch.sin(theta_total)     # (K,)
        cos_t = torch.cos(theta_total)     # (K,)
        sin_p = torch.sin(phi)             # (K,)
        cos_p = torch.cos(phi)             # (K,)

        # Combine into world-frame vector: (3, K)
        bond = (
            e_x * (sin_t * cos_p).unsqueeze(0) +
            e_y * (sin_t * sin_p).unsqueeze(0) +
            e_z * cos_t.unsqueeze(0)
        )

        # Normalize for numerical safety (should already be unit length)
        return bond / torch.linalg.norm(bond, dim=0, keepdim=True)

    @staticmethod
    def reconstruct_torsion(
        next_bond: torch.Tensor,
        curr_bond: torch.Tensor,
        prev_bond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct torsion angle from three consecutive bond vectors.

        Computes the dihedral angle between the plane (prev_bond, curr_bond)
        and the plane (curr_bond, next_bond), which is exactly the torsion
        angle τ that TorsionMCMC would have sampled to produce next_bond.

        This allows PolymerAnalyzer to compute torsion statistics on
        Fisher-generated chains identically to Von Mises chains.

        Args:
            next_bond: Incoming bond vector, shape (3, K)
            curr_bond: Current bond vector,  shape (3, K)
            prev_bond: Previous bond vector, shape (3, K)

        Returns:
            Torsion angles in [-π, π], shape (K,)
        """
        # Normal to plane (prev → curr)
        n1 = torch.linalg.cross(prev_bond, curr_bond, dim=0)
        n1 = n1 / torch.linalg.norm(n1, dim=0, keepdim=True).clamp(min=1e-8)

        # Normal to plane (curr → next)
        n2 = torch.linalg.cross(curr_bond, next_bond, dim=0)
        n2 = n2 / torch.linalg.norm(n2, dim=0, keepdim=True).clamp(min=1e-8)

        # Torsion reference vector (in-plane perpendicular to curr_bond)
        curr_norm = curr_bond / torch.linalg.norm(
            curr_bond, dim=0, keepdim=True
        ).clamp(min=1e-8)
        m1 = torch.linalg.cross(n1, curr_norm, dim=0)

        # atan2 for signed angle
        cos_tau = (n1 * n2).sum(dim=0)           # (K,)
        sin_tau = (m1 * n2).sum(dim=0)            # (K,)

        return torch.atan2(sin_tau, cos_tau)       # (K,)


# =============================================================================
# Abstract base sampler
# =============================================================================

class BaseSampler(ABC):
    """
    Abstract base class for polymer torsion/bond samplers.

    Defines the shared interface and pS parameterization used by
    both VonMisesSampler (S¹) and FisherSampler (S²).
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.device = config.torch_device
        self.L = config.total_length

        # Shared pS grid
        self.pS_grid = torch.linspace(
            config.pS_range[0],
            config.pS_range[1],
            config.pS_res,
            device=self.device,
            dtype=torch.float64
        )
        self.kappa_grid = 10 ** (7 - self.pS_grid)

        self.p_grid = torch.linspace(
            0, 1,
            config.p_res,
            device=self.device,
            dtype=torch.float64
        )

        self.cache = GridCache(config.cache_dir)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Generate samples."""
        ...


# =============================================================================
# Von Mises Sampler (S¹) — unchanged from original
# =============================================================================

class VonMisesSampler(BaseSampler):
    """
    Efficient sampler for Von Mises mixture distributions on S¹.

    Uses bicubic interpolation on precomputed ICDF grids to generate
    torsion angle samples. Supports batched sampling across multiple
    concentration parameters simultaneously.

    Output: scalar torsion angles, shape (1, 2^pL, K)

    This is the degenerate limit of FisherSampler when κ_θ → ∞
    (bond angle frozen at tetrahedral value). Use for:
        - Lightweight computation
        - Validating FisherSampler output
        - Cases where bond angle disorder is negligible
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.tau_c = config.tau_centers_radians

    def _get_or_create_grid(
        self,
        weights: Tuple[float, float, float]
    ) -> torch.Tensor:
        """Load ICDF grid from cache or compute if needed."""
        grid = self.cache.load(
            class_name=self.__class__.__name__,
            weights=weights,
            pS_range=self.config.pS_range,
            pS_res=self.config.pS_res,
            p_res=self.config.p_res
        )

        if grid is not None:
            return grid.to(self.device)

        print(f"⚙️  Computing new ICDF grid for weights={weights}")
        grid_np = VonMisesDistribution.create_icdf_grid(
            weights=weights,
            mus=self.tau_c,
            k_grid=self.kappa_grid.cpu().numpy(),
            p_grid=self.p_grid.cpu().numpy(),
            verbose=True
        )

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
        Bicubic interpolation on (pS × probability) ICDF grid.

        Args:
            p_samples: Uniform probabilities, shape (1, L, K)
            pS_values: Entropy parameters,    shape (K,)
            icdf_grid: ICDF lookup table

        Returns:
            Torsion angle samples, shape (1, L, K)
        """
        K = pS_values.shape[0]
        grid_4d = icdf_grid.view(1, 1, *icdf_grid.shape)

        pS_min, pS_max = self.pS_grid[0], self.pS_grid[-1]
        pS_norm = 2.0 * (pS_values - pS_min) / (pS_max - pS_min) - 1.0

        p_min, p_max = self.p_grid[0], self.p_grid[-1]
        p_norm = 2.0 * (p_samples - p_min) / (p_max - p_min) - 1.0

        pS_norm_expanded = pS_norm.view(1, 1, K).expand(1, self.L, K)
        sampling_grid = torch.stack([p_norm, pS_norm_expanded], dim=-1)
        sampling_grid = sampling_grid.view(1, self.L, K, 2).to(torch.float32)

        samples = F.grid_sample(
            grid_4d,
            sampling_grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True
        )

        return samples.view(1, self.L, K).to(torch.float64)

    def __call__(
        self,
        weights: Tuple[float, float, float],
        pS_values: list[float] | np.ndarray | torch.Tensor = None,
        kappas: list[float] | np.ndarray | torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample torsion angles from Von Mises mixture on S¹.

        Args:
            weights:   Mixture weights (T, Gp, Gn)
            pS_values: Entropy parameters, length K  (preferred)
            kappas:    Concentration parameters, length K (legacy)

        Returns:
            Torsion angles, shape (1, 2^pL, K)
        """
        if pS_values is None and kappas is None:
            raise ValueError("Must provide either pS_values or kappas")
        if pS_values is not None and kappas is not None:
            raise ValueError("Provide either pS_values OR kappas, not both")

        if kappas is not None:
            pS_values = self.config.kappa_to_ps(np.asarray(kappas))

        icdf_grid = self._get_or_create_grid(weights)

        pS_tensor = torch.as_tensor(pS_values, device=self.device, dtype=torch.float64)
        K = pS_tensor.shape[0]

        p_samples = torch.rand(
            (1, self.L, 1), device=self.device, dtype=torch.float64
        ).expand(-1, -1, K)

        return self._interpolate_2d(p_samples, pS_tensor, icdf_grid)

    def get_pS_gradient(
        self,
        weights: Tuple[float, float, float],
        pS_values: list[float] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ∂τ/∂pS via autograd through the bicubic interpolation.

        Args:
            weights:   Mixture weights
            pS_values: Entropy parameters

        Returns:
            Gradients ∂τ/∂pS, shape (1, L, K)
        """
        pS_tensor = torch.as_tensor(
            pS_values, device=self.device, dtype=torch.float64
        )
        pS_tensor.requires_grad_(True)

        icdf_grid = self._get_or_create_grid(weights)
        p_samples = torch.rand(
            (1, self.L, 1), device=self.device, dtype=torch.float64
        ).expand(-1, -1, pS_tensor.shape[0])

        taus = self._interpolate_2d(p_samples, pS_tensor, icdf_grid)

        return torch.autograd.grad(
            outputs=taus,
            inputs=pS_tensor,
            grad_outputs=torch.ones_like(taus),
            create_graph=True
        )[0]

    def get_kappa_gradient(
        self,
        weights: Tuple[float, float, float],
        kappas: list[float] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ∂τ/∂κ via chain rule through ∂τ/∂pS.

        Args:
            weights: Mixture weights
            kappas:  Concentration parameters

        Returns:
            Gradients ∂τ/∂κ, shape (1, L, K)
        """
        kappas_array = np.asarray(kappas)
        dτ_dpS = self.get_pS_gradient(weights, self.config.kappa_to_ps(kappas_array))

        kappas_tensor = torch.as_tensor(
            kappas_array, device=self.device, dtype=torch.float64
        )
        dpS_dκ = (-1 / (kappas_tensor * np.log(10))).view(1, 1, -1)

        return dτ_dpS * dpS_dκ


# =============================================================================
# Fisher Sampler (S²)
# =============================================================================

class FisherSampler(BaseSampler):
    """
    Stateful sampler for mixtures of von Mises–Fisher distributions on S².

    Generalizes VonMisesSampler from S¹ to S² by introducing a second
    concentration parameter κ_θ controlling the polar (bond angle) spread.

    Parameterization:
        κ_φ  — azimuthal / torsional concentration  (same role as Von Mises κ)
        κ_θ  — polar / bond-angle concentration

    Limits:
        κ_θ → ∞  :  bond angle frozen at tetrahedral value
                    Fisher on S² → Von Mises on S¹  (validation case)
        κ_θ → 0  :  uniform polar sampling (freely-jointed chain limit)
        κ_φ → 0  :  uniform azimuthal sampling (no torsional preference)

    The three mixture components are placed at Trans/Gauche+/Gauche- azimuths
    at the tetrahedral polar angle from curr_bond, directly generalizing the
    VonMisesSampler's three peaks from S¹ onto S².

    Statefulness:
        The sampler is called per-step by FisherMCMC, receiving the current
        local frame (curr_bond, prev_bond) which defines the mean directions
        of all three mixture components. The sampler itself is stateless —
        all state lives in FisherMCMC.

    Output per step: (3, K) unit bond vectors
    Output over full chain (via FisherMCMC): (3, 2^pL, K)
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize Fisher sampler.

        Args:
            config: Simulation configuration. Uses:
                - pL:              total chain length parameter
                - pS_range:        used for azimuthal (φ) kappa grid
                - p_res, pS_res:   grid resolution
                - tau_centers:     conformer azimuths (Trans, G+, G-)
                - fisher_bond_angle: tetrahedral offset (arccos of alignment)
                - cache_dir:       for azimuthal ICDF cache
        """
        super().__init__(config)
        self.tau_c = config.tau_centers_radians
        self.bond_angle = config.fisher_bond_angle  # polar offset in radians

        # Pre-build the azimuthal ICDF grid lookup arrays (numpy, for per-step use)
        # These are keyed by weights and loaded/computed on first call to
        # _get_or_create_azimuthal_grid()
        self._azimuthal_grids: dict = {}

    def _get_or_create_azimuthal_grid(
        self,
        weights: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load or compute the azimuthal (Von Mises) ICDF grid.

        Returns numpy arrays rather than tensors since the Fisher sampler
        uses them in a per-step loop rather than a batched tensor operation.

        Args:
            weights: Mixture weights (T, Gp, Gn)

        Returns:
            (kappa_grid_np, icdf_grid_np) as numpy arrays
            kappa_grid_np: shape (pS_res,)
            icdf_grid_np:  shape (pS_res, p_res)
        """
        key = weights
        if key in self._azimuthal_grids:
            return self._azimuthal_grids[key]

        # Try cache (reuses VonMisesSampler cache key format)
        grid_tensor = self.cache.load(
            class_name="VonMisesSampler",  # shared cache with VonMisesSampler
            weights=weights,
            pS_range=self.config.pS_range,
            pS_res=self.config.pS_res,
            p_res=self.config.p_res
        )

        if grid_tensor is None:
            print(f"⚙️  Computing azimuthal ICDF grid for Fisher sampler, weights={weights}")
            grid_np = VonMisesDistribution.create_icdf_grid(
                weights=weights,
                mus=self.tau_c,
                k_grid=self.kappa_grid.cpu().numpy(),
                p_grid=self.p_grid.cpu().numpy(),
                verbose=True
            )
            self.cache.save(
                grid_np,
                class_name="VonMisesSampler",
                weights=weights,
                pS_range=self.config.pS_range,
                pS_res=self.config.pS_res,
                p_res=self.config.p_res
            )
        else:
            grid_np = grid_tensor.numpy()

        kappa_np = self.kappa_grid.cpu().numpy()
        p_np = self.p_grid.cpu().numpy()

        self._azimuthal_grids[key] = (kappa_np, p_np, grid_np)
        return kappa_np, p_np, grid_np

    def _kappa_to_grid_row(
        self,
        kappa_phi: np.ndarray,
        kappa_grid_np: np.ndarray,
        icdf_grid_np: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate ICDF grid rows for given kappa_phi values.

        For each kappa in kappa_phi, returns the corresponding ICDF row
        (a function of probability) by linear interpolation between
        the nearest grid rows.

        Args:
            kappa_phi:    Target kappa values, shape (K,)
            kappa_grid_np: Grid kappa axis,    shape (pS_res,)
            icdf_grid_np:  Full ICDF grid,     shape (pS_res, p_res)

        Returns:
            Interpolated ICDF rows, shape (K, p_res)
        """
        K = len(kappa_phi)
        p_res = icdf_grid_np.shape[1]
        rows = np.zeros((K, p_res))

        for k_idx in range(K):
            # Find bracketing indices in kappa_grid
            # kappa_grid is DESCENDING (high kappa = low pS = index 0)
            # Use log-space interpolation for kappa
            log_k = np.log10(np.clip(kappa_phi[k_idx], 1e-12, None))
            log_grid = np.log10(np.clip(kappa_grid_np, 1e-12, None))

            # kappa_grid is descending, so flip for searchsorted
            idx = np.searchsorted(-log_grid, -log_k)
            idx = np.clip(idx, 1, len(kappa_grid_np) - 1)

            # Linear interpolation in log-kappa space
            lo, hi = idx - 1, idx
            t = (log_k - log_grid[lo]) / (log_grid[hi] - log_grid[lo] + 1e-12)
            t = np.clip(t, 0.0, 1.0)
            rows[k_idx] = (1 - t) * icdf_grid_np[lo] + t * icdf_grid_np[hi]

        return rows

    def step(
        self,
        curr_bond: torch.Tensor,
        prev_bond: torch.Tensor,
        kappa_theta: torch.Tensor,
        kappa_phi: torch.Tensor,
        weights: Tuple[float, float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one next bond vector from the Fisher mixture.

        Constructs the local frame from (curr_bond, prev_bond),
        samples (θ, φ) from their respective marginals, converts to
        a world-frame unit vector, and reconstructs the torsion angle.

        Args:
            curr_bond:   Current bond direction, shape (3, K)
            prev_bond:   Previous bond direction, shape (3, K)
            kappa_theta: Polar concentration,     shape (K,)
            kappa_phi:   Azimuthal concentration, shape (K,)
            weights:     Mixture weights (T, Gp, Gn)

        Returns:
            next_bond: Unit bond vectors,   shape (3, K)
            tau:       Torsion angles (reconstructed), shape (K,)
        """
        K = curr_bond.shape[1]

        # ------------------------------------------------------------------
        # 1. Build local orthonormal frame
        # ------------------------------------------------------------------
        e_z, e_x, e_y = FisherDistribution.build_local_frame(curr_bond, prev_bond)

        # ------------------------------------------------------------------
        # 2. Sample polar angle θ from Fisher marginal (closed-form ICDF)
        # ------------------------------------------------------------------
        kappa_theta_np = kappa_theta.cpu().numpy()
        theta_np = FisherDistribution.sample_polar(kappa_theta_np, n=1)  # (1, K)
        theta = torch.as_tensor(
            theta_np[0], device=self.device, dtype=torch.float64
        )  # (K,)

        # ------------------------------------------------------------------
        # 3. Sample azimuthal angle φ from Von Mises mixture
        # ------------------------------------------------------------------
        kappa_np, p_np, icdf_np = self._get_or_create_azimuthal_grid(weights)
        kappa_phi_np = kappa_phi.cpu().numpy()

        # Get interpolated ICDF rows for each kappa_phi value: (K, p_res)
        icdf_rows = self._kappa_to_grid_row(kappa_phi_np, kappa_np, icdf_np)

        # Sample uniform probabilities and invert via ICDF rows
        p_samples = np.random.uniform(0.0, 1.0, size=K)
        phi_np = np.array([
            np.interp(p_samples[k], p_np, icdf_rows[k]) for k in range(K)
        ])
        phi = torch.as_tensor(phi_np, device=self.device, dtype=torch.float64)  # (K,)

        # ------------------------------------------------------------------
        # 4. Convert (θ, φ) → world-frame unit bond vector
        # ------------------------------------------------------------------
        next_bond = FisherDistribution.spherical_to_world(
            theta, phi, e_z, e_x, e_y, self.bond_angle
        )  # (3, K)

        # ------------------------------------------------------------------
        # 5. Reconstruct torsion angle for PolymerAnalyzer compatibility
        # ------------------------------------------------------------------
        tau = FisherDistribution.reconstruct_torsion(next_bond, curr_bond, prev_bond)

        return next_bond, tau

    def __call__(
        self,
        weights: Tuple[float, float, float],
        curr_bond: torch.Tensor,
        prev_bond: torch.Tensor,
        kappa_theta: torch.Tensor,
        kappa_phi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Callable interface for FisherMCMC.

        Wraps step() so FisherMCMC can call the sampler uniformly.

        Args:
            weights:     Mixture weights (T, Gp, Gn)
            curr_bond:   Current bond direction, shape (3, K)
            prev_bond:   Previous bond direction, shape (3, K)
            kappa_theta: Polar concentration,     shape (K,)
            kappa_phi:   Azimuthal concentration, shape (K,)

        Returns:
            next_bond: shape (3, K)
            tau:       shape (K,)
        """
        return self.step(curr_bond, prev_bond, kappa_theta, kappa_phi, weights)