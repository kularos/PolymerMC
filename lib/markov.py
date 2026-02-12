"""
Markov Chain Monte Carlo for polymer chain generation.

This module implements the MCMC algorithm for generating polymer chains
from torsion angle samples using bond rotation mechanics.
"""

import torch
from typing import Optional

from .config import SimulationConfig
from .core import rodrigues_rotation


class TorsionMCMC:
    """
    Monte Carlo Markov Chain polymer chain generator.

    Builds 3D polymer chains by iteratively applying torsion rotations
    to bond vectors. Each monomer is added by rotating the previous bond
    around the current bond axis by the sampled torsion angle.

    The algorithm:
    1. Initialize with two initial bond vectors
    2. For each torsion angle sample:
       - Rotate previous bond around current bond
       - Add new monomer at rotated position
       - Update bond vectors for next iteration

    Attributes:
        chain_length: Number of monomers per chain
        n_chains: Number of independent chains
        n_batches: Number of parameter sets (e.g., different kappa values)
        shape: Full tensor shape (3, chain_length, n_chains, n_batches)
    """

    def __init__(
            self,
            config: SimulationConfig,
            n_batches: int = 1
    ):
        """
        Initialize MCMC chain builder.

        Args:
            config: Simulation configuration
            n_batches: Number of parameter batches to process simultaneously
        """
        self._device = config.torch_device
        self.chain_length = config.chain_length
        self.n_chains = config.n_chains
        self.n_batches = n_batches

        # Tensor shapes for batched operations
        # Vectors: (3, n_chains, n_batches)
        # Full chains: (3, chain_length, n_chains, n_batches)
        self.batch_vec_shape = (3, self.n_chains, n_batches)
        self.shape = (3, self.chain_length, self.n_chains, self.n_batches)

        # State variables
        self.prev_bond: Optional[torch.Tensor] = None
        self.curr_bond: Optional[torch.Tensor] = None
        self._chains: Optional[torch.Tensor] = None
        self._monomer_index: int = 0

    def init_chains(self, alignment: float = 1 / 3) -> None:
        """
        Initialize polymer chains with starting configuration.

        Sets up the first three monomers:
        - Monomer 0: at -prev_bond (behind origin)
        - Monomer 1: at origin
        - Monomer 2: at +curr_bond (ahead)

        Args:
            alignment: Controls initial bond angle via z-component.
                      alignment=1 → parallel bonds (0°)
                      alignment=0 → perpendicular bonds (90°)
                      alignment=1/3 → tetrahedral angle (~109.5°)
        """
        # Initial bond vectors
        # prev_bond points from monomer 0 to 1
        # curr_bond points from monomer 1 to 2
        y_component = (1 - alignment ** 2) ** 0.5

        init_prev = torch.tensor(
            [0.0, y_component, alignment],
            device=self._device,
            dtype=torch.float64
        )
        init_curr = torch.tensor(
            [0.0, 0.0, 1.0],
            device=self._device,
            dtype=torch.float64
        )

        # Expand to batch dimensions: (3, 1, 1) -> (3, N, K)
        self.prev_bond = init_prev.view(3, 1, 1).expand(self.batch_vec_shape)
        self.curr_bond = init_curr.view(3, 1, 1).expand(self.batch_vec_shape)

        # Allocate chain storage: (3, L, N, K)
        self._chains = torch.zeros(
            self.shape,
            device=self._device,
            dtype=torch.float64
        )

        # Place first three monomers
        self._chains[:, 0, :, :] = -self.prev_bond  # Behind origin
        # Monomer 1 at origin (already zeros)
        self._chains[:, 2, :, :] = self.curr_bond  # Ahead of origin

        self._monomer_index = 1

    def step(self, torsion_angle: torch.Tensor) -> None:
        """
        Add one monomer to each chain via torsion rotation.

        Rotates the previous bond vector around the current bond axis
        by the given torsion angle, then adds the new monomer at the
        resulting position.

        Args:
            torsion_angle: Rotation angles, shape (n_chains, n_batches)
        """
        # Rotate previous bond around current bond axis
        next_bond = rodrigues_rotation(
            self.prev_bond,
            self.curr_bond,
            torsion_angle,
            vector_dim=0
        )

        # Place new monomer relative to current position
        self._chains[:, self._monomer_index + 1] = (
                self._chains[:, self._monomer_index] + next_bond
        )

        # Update bond vectors for next iteration
        self.prev_bond = self.curr_bond
        self.curr_bond = next_bond
        self._monomer_index += 1

    def run(self, torsion_samples: torch.Tensor) -> None:
        """
        Generate complete polymer chains from torsion angle samples.

        Args:
            torsion_samples: Torsion angles, shape (n_batches, n_chains, L)
                            or (n_batches, n_chains, L, 1)

        Notes:
            Automatically initializes chains and iterates through all
            torsion angles to build complete polymer configurations.
        """
        # Handle optional trailing dimension
        if torsion_samples.dim() == 4:
            torsion_samples = torsion_samples.squeeze(-1)

        # Initialize starting configuration
        self.init_chains()

        # Build chains monomer by monomer
        for i in range(self._monomer_index, self.chain_length - 1):
            # Extract torsion angles for this step: (n_batches, n_chains)
            tau_i = torsion_samples[:, i]
            self.step(tau_i)

    @property
    def chains(self) -> torch.Tensor:
        """
        Get generated polymer chains.

        Returns:
            Chain coordinates, shape (3, chain_length, n_chains, n_batches)
        """
        return self._chains

    def reset(self) -> None:
        """Reset the MCMC state for a new run."""
        self.prev_bond = None
        self.curr_bond = None
        self._chains = None
        self._monomer_index = 0
