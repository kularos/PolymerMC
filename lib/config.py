"""
Configuration management for polymer MCMC simulation.

This module provides a centralized configuration class that manages
all simulation parameters, RNG seeding, device allocation, and paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import torch
import numpy as np
import random


@dataclass
class SimulationConfig:
    """
    Central configuration for polymer Monte Carlo simulation.

    This class manages:
    - Simulation parameters (hierarchical bisection via pL and pGamma)
    - Hardware configuration (CPU/GPU device)
    - Random number generator seeding
    - Sampler parameters (concentration ranges, resolution)
    - File paths (cache, output directories)
    - Visualization settings

    Hierarchical Bisection Structure:
        The simulation generates ONE long chain of length 2^pL, which can be
        analyzed at different subdivision levels via pGamma:
        - pL = log2(total_length): Total monomers in the full chain
        - pGamma = log2(n_bisections): How many times to subdivide

        Example (pL=12, total=4096):
            pGamma=0: 1 chain × 4096 monomers
            pGamma=1: 2 chains × 2048 monomers
            pGamma=2: 4 chains × 1024 monomers
            pGamma=3: 8 chains × 512 monomers

    Attributes:
        pL: log2 of total chain length (pL=7 → 128, pL=12 → 4096)
        pGamma: log2 of number of bisections (pGamma=3 → 8 chains)
        seed: Random seed for reproducibility (None = random seed)
        device: Device for PyTorch tensors ("cpu" or "cuda")
        pS_range: (min, max) entropy parameter range for Von Mises
        pS_res: Number of pS values in grid
        p_res: Number of probability values in CDF grid
        tau_centers: Torsion angle centers in degrees (Trans, Gauche+, Gauche-)
        cache_dir: Directory for caching ICDF lookup tables
        output_dir: Directory for simulation outputs (GIFs, plots)
        frame_dpi: DPI resolution for animation frames
        gif_duration: Duration per frame in milliseconds
        gif_fps: Frames per second for animations
    """

    # Core simulation parameters
    # pGamma is no longer a config-level field — bisection level is chosen
    # lazily at analysis time via PolymerAnalyzer.generate_assets(pGamma).
    pL: int = 7  # log2(total_length): pL=7 → 128 monomers, pL=12 → 4096 monomers
    seed: int | None = 1738

    # Hardware configuration
    device: str = "cpu"

    # Von Mises sampler parameters
    # pS parameterization: pS = 7 - log10(κ)
    # Higher pS = higher entropy (more flexible chains)
    # pS=5 → κ=10², pS=7 → κ=1, pS=9 → κ=10⁻²
    pS_range: Tuple[float, float] = (5.0, 9.0)  # Entropy range
    pS_res: int = 1000  # Resolution of pS grid
    p_res: int = 100000  # Probability resolution for ICDF
    tau_centers: Tuple[float, float, float] = (0.0, -120.0, 120.0)  # degrees

    # File system paths
    cache_dir: Path = field(default_factory=lambda: Path("./local/cache/ICDF_cache"))
    output_dir: Path = field(default_factory=lambda: Path("./local/outputs/gifs"))

    # Visualization parameters
    frame_dpi: int = 256
    gif_duration: int = 20  # milliseconds per frame
    gif_fps: int = 50

    # Private fields (computed in __post_init__)
    _torch_device: torch.device = field(init=False, repr=False)
    _seed_applied: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Initialize derived properties and create directories."""
        # Convert string paths to Path objects
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize torch device
        self._torch_device = torch.device(self.device)

        # Apply seed if provided
        if self.seed is not None and not self._seed_applied:
            self.apply_seed()

    def apply_seed(self) -> None:
        """
        Apply RNG seed to all random number generators.

        This ensures reproducibility across:
        - Python's random module
        - NumPy's random module
        - PyTorch CPU and CUDA operations
        """
        if self.seed is None:
            print("⚠️  No seed set - using random initialization")
            return

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._seed_applied = True
        print(f"🌱 Global seed set to: {self.seed}")

    @property
    def torch_device(self) -> torch.device:
        """Get PyTorch device object for tensor allocation."""
        return self._torch_device

    @property
    def total_length(self) -> int:
        """
        Get total number of monomers from pL.

        Returns:
            Total chain length = 2^pL

        Example:
            >>> config = SimulationConfig(pL=10)
            >>> config.total_length
            1024
        """
        return 2 ** self.pL

    @property
    def valid_pGamma_range(self) -> range:
        """
        All valid bisection levels for this configuration: 0 .. pL inclusive.

            pGamma=0:  1 chain  × 2^pL monomers  (full chain view)
            pGamma=pL: 2^pL chains × 1 monomer   (monomer scale)
        """
        return range(0, self.pL + 1)

    @property
    def tau_centers_radians(self) -> np.ndarray:
        """Get torsion angle centers converted to radians."""
        return np.radians(self.tau_centers)

    @property
    def kappa_range(self) -> Tuple[float, float]:
        """
        Get kappa range from pS range.

        Returns:
            (kappa_min, kappa_max) corresponding to (pS_max, pS_min)
            Note the reversal: higher pS = lower kappa
        """
        pS_min, pS_max = self.pS_range
        kappa_max = 10 ** (7 - pS_min)  # Low entropy = high kappa
        kappa_min = 10 ** (7 - pS_max)  # High entropy = low kappa
        return (kappa_min, kappa_max)

    @staticmethod
    def pS_to_kappa(ps: float | np.ndarray) -> float | np.ndarray:
        """
        Convert pS (entropy parameter) to κ (concentration).

        Args:
            ps: Entropy parameter (higher = more flexible)

        Returns:
            kappa: Concentration parameter (higher = more stiff)

        Example:
            >>> SimulationConfig.pS_to_kappa(5.0)
            100.0  # Flexible
            >>> SimulationConfig.pS_to_kappa(9.0)
            0.01   # Stiff
        """
        return 10 ** (7 - ps)

    @staticmethod
    def kappa_to_ps(kappa: float | np.ndarray) -> float | np.ndarray:
        """
        Convert κ (concentration) to pS (entropy parameter).

        Args:
            kappa: Concentration parameter

        Returns:
            ps: Entropy parameter

        Example:
            >>> SimulationConfig.kappa_to_ps(100.0)
            5.0
            >>> SimulationConfig.kappa_to_ps(0.01)
            9.0
        """
        return 7 - np.log10(kappa)

    def output_path(self, weights: Tuple[float, float, float]) -> Path:
        """
        Get output directory path for specific weight configuration.

        Args:
            weights: (T, Gp, Gn) weight tuple

        Returns:
            Path to weight-specific output directory
        """
        T, Gp, Gn = weights
        path = self.output_dir / f"W({T},{Gp},{Gn})"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self) -> str:
        """Formatted string representation of configuration."""
        return (
            f"SimulationConfig(\n"
            f"  pL={self.pL}, total_length={self.total_length}\n"
            f"  pGamma sweep: 0 .. {self.pL} ({self.pL + 1} levels)\n"
            f"  seed={self.seed}, device={self.device}\n"
            f"  pS_range={self.pS_range}, pS_res={self.pS_res}\n"
            f"  cache_dir={self.cache_dir}\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )