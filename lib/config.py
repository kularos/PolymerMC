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
    - Simulation parameters (chain length, number of chains)
    - Hardware configuration (CPU/GPU device)
    - Random number generator seeding
    - Sampler parameters (concentration ranges, resolution)
    - File paths (cache, output directories)
    - Visualization settings

    Attributes:
        chain_length: Number of monomers in each polymer chain
        n_chains: Number of independent polymer chains to simulate
        seed: Random seed for reproducibility (None = random seed)
        device: Device for PyTorch tensors ("cpu" or "cuda")
        k_range: (min, max) concentration parameter range for Von Mises
        k_res: Number of kappa values in logarithmic grid
        p_res: Number of probability values in CDF grid
        tau_centers: Torsion angle centers in degrees (Trans, Gauche+, Gauche-)
        cache_dir: Directory for caching ICDF lookup tables
        output_dir: Directory for simulation outputs (GIFs, plots)
        frame_dpi: DPI resolution for animation frames
        gif_duration: Duration per frame in milliseconds
        gif_fps: Frames per second for animations
    """

    # Core simulation parameters
    chain_length: int = 128
    n_chains: int = 8
    seed: int | None = 1738

    # Hardware configuration
    device: str = "cpu"

    # Von Mises sampler parameters
    k_range: Tuple[float, float] = (1e-3, 1e7)
    k_res: int = 1000
    p_res: int = 100000
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
    def tau_centers_radians(self) -> np.ndarray:
        """Get torsion angle centers converted to radians."""
        return np.radians(self.tau_centers)

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
            f"  chain_length={self.chain_length}, n_chains={self.n_chains}\n"
            f"  seed={self.seed}, device={self.device}\n"
            f"  k_range={self.k_range}, k_res={self.k_res}\n"
            f"  cache_dir={self.cache_dir}\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )
