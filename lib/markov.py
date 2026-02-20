"""
Markov Chain Monte Carlo for polymer chain generation.

Provides two MCMC chain builders on the manifold hierarchy:
  - TorsionMCMC : driven by scalar torsion angles from VonMisesSampler (S¹)
  - FisherMCMC  : driven by 3D bond vectors from FisherSampler (S²)

Both produce chains of shape (3, 2^pL, K) and torsion samples of shape
(2^pL, K), making them drop-in compatible with PolymerAnalyzer.

Class hierarchy:
    BaseMCMC (abstract)
    ├── TorsionMCMC   driven by VonMisesSampler — unchanged from original
    └── FisherMCMC    driven by FisherSampler
                      owns sequential loop, reconstructs torsion per step
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from .config import SimulationConfig
from .core import rodrigues_rotation


# =============================================================================
# Abstract base MCMC
# =============================================================================

class BaseMCMC(ABC):
    """
    Abstract base class for polymer MCMC chain builders.

    Manages chain storage, initialization, and the shared monomer-placement
    logic. Subclasses implement step() and run() for their specific
    sampling geometry.

    Attributes:
        total_length: Total number of monomers (2^pL)
        n_batches:    Number of parameter batches (K — e.g. pS values)
        shape:        Full chain tensor shape (3, total_length, K)
    """

    def __init__(
        self,
        config: SimulationConfig,
        n_batches: int = 1
    ):
        """
        Initialize MCMC chain builder.

        Args:
            config:    Simulation configuration (uses pL for total length)
            n_batches: Number of parameter batches K
        """
        self._device = config.torch_device
        self.total_length = config.total_length  # 2^pL
        self.n_batches = n_batches
        self.alignment = config.fisher_bond_angle_alignment  # alignment param

        self.batch_vec_shape = (3, n_batches)
        self.shape = (3, self.total_length, n_batches)

        self.prev_bond: Optional[torch.Tensor] = None
        self.curr_bond: Optional[torch.Tensor] = None
        self._chains: Optional[torch.Tensor] = None
        self._taus: Optional[torch.Tensor] = None    # reconstructed torsions
        self._monomer_index: int = 0

    def init_chains(self) -> None:
        """
        Initialize chain storage and starting bond configuration.

        Places first three monomers:
            monomer 0: at -prev_bond  (behind origin)
            monomer 1: at origin
            monomer 2: at +curr_bond  (ahead)

        The initial bond angle is set by config.fisher_bond_angle_alignment:
            alignment=1   → parallel bonds (0°)
            alignment=0   → perpendicular (90°)
            alignment=1/3 → tetrahedral (~109.5°)
        """
        a = self.alignment
        y_component = (1 - a ** 2) ** 0.5

        init_prev = torch.tensor(
            [0.0, y_component, a],
            device=self._device, dtype=torch.float64
        )
        init_curr = torch.tensor(
            [0.0, 0.0, 1.0],
            device=self._device, dtype=torch.float64
        )

        self.prev_bond = init_prev.view(3, 1).expand(self.batch_vec_shape).clone()
        self.curr_bond = init_curr.view(3, 1).expand(self.batch_vec_shape).clone()

        self._chains = torch.zeros(
            self.shape, device=self._device, dtype=torch.float64
        )
        self._taus = torch.zeros(
            (self.total_length, self.n_batches),
            device=self._device, dtype=torch.float64
        )

        self._chains[:, 0, :] = -self.prev_bond
        # monomer 1 at origin (already zeros)
        self._chains[:, 2, :] = self.curr_bond

        self._monomer_index = 1

    def _place_monomer(self, next_bond: torch.Tensor) -> None:
        """
        Place next monomer using the given bond vector.

        Args:
            next_bond: Bond vector, shape (3, K) — should be unit length
        """
        self._chains[:, self._monomer_index + 1] = (
            self._chains[:, self._monomer_index] + next_bond
        )

    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Advance chain by one monomer."""
        ...

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """Generate complete chain."""
        ...

    @property
    def chains(self) -> torch.Tensor:
        """
        Generated chain coordinates, shape (3, total_length, K).

        Drop-in compatible with PolymerAnalyzer input.
        """
        return self._chains

    @property
    def torsion_samples(self) -> torch.Tensor:
        """
        Torsion angle samples, shape (1, total_length, K).

        For TorsionMCMC: the sampled angles directly.
        For FisherMCMC:  torsion angles reconstructed from consecutive bonds.

        Shape matches VonMisesSampler output for PolymerAnalyzer compatibility.
        """
        return self._taus.unsqueeze(0)  # (1, total_length, K)

    def reset(self) -> None:
        """Reset MCMC state for a new run."""
        self.prev_bond = None
        self.curr_bond = None
        self._chains = None
        self._taus = None
        self._monomer_index = 0


# =============================================================================
# Torsion MCMC (S¹) — driven by VonMisesSampler, unchanged from original
# =============================================================================

class TorsionMCMC(BaseMCMC):
    """
    MCMC chain builder driven by scalar torsion angles (VonMisesSampler).

    Builds a 3D polymer chain by rotating the previous bond around the
    current bond axis by the sampled torsion angle at each step. Bond
    length and bond angle are preserved exactly (Rodrigues rotation).

    This is the original TorsionMCMC, now inheriting from BaseMCMC.
    All behavior is identical to the previous version.
    """

    def step(self, torsion_angle: torch.Tensor) -> None:
        """
        Add one monomer via torsion rotation.

        Args:
            torsion_angle: Rotation angles, shape (K,)
        """
        next_bond = rodrigues_rotation(
            self.prev_bond,
            self.curr_bond,
            torsion_angle,
            vector_dim=0
        )

        self._place_monomer(next_bond)
        self._taus[self._monomer_index] = torsion_angle

        self.prev_bond = self.curr_bond
        self.curr_bond = next_bond
        self._monomer_index += 1

    def run(self, torsion_samples: torch.Tensor) -> None:
        """
        Generate complete chain from pre-sampled torsion angles.

        Args:
            torsion_samples: Torsion angles, shape (1, 2^pL, K)
        """
        if torsion_samples.dim() == 3 and torsion_samples.shape[0] == 1:
            torsion_samples = torsion_samples.squeeze(0)  # (2^pL, K)

        self.init_chains()

        for i in range(self._monomer_index, self.total_length - 1):
            self.step(torsion_samples[i])


# =============================================================================
# Fisher MCMC (S²) — driven by FisherSampler
# =============================================================================

class FisherMCMC(BaseMCMC):
    """
    MCMC chain builder driven by 3D bond vectors from FisherSampler.

    At each step, the FisherSampler is called with the current local frame
    (curr_bond, prev_bond) and returns a new unit bond vector sampled from
    the Fisher mixture on S². The torsion angle is reconstructed from
    consecutive bond vectors and stored for PolymerAnalyzer compatibility.

    The run() method owns the sequential loop and is responsible for
    passing the evolving local frame state to the sampler — this is what
    makes the chain stateful.

    Output shapes match TorsionMCMC exactly:
        chains:          (3, 2^pL, K)
        torsion_samples: (1, 2^pL, K)  ← reconstructed torsion angles

    Validation:
        Set kappa_theta to a large value (e.g. 1e6) and compare torsion
        histograms with TorsionMCMC at the same kappa_phi — they should
        be numerically indistinguishable.
    """

    def step(
        self,
        next_bond: torch.Tensor,
        tau: torch.Tensor,
    ) -> None:
        """
        Add one monomer using a pre-sampled bond vector.

        Unlike TorsionMCMC.step(), the bond vector is computed externally
        by FisherSampler and passed in directly. This keeps the sampling
        logic in the sampler and the chain-building logic in the MCMC.

        Args:
            next_bond: Unit bond vector, shape (3, K)
            tau:       Reconstructed torsion angle, shape (K,)
        """
        self._place_monomer(next_bond)
        self._taus[self._monomer_index] = tau

        self.prev_bond = self.curr_bond
        self.curr_bond = next_bond
        self._monomer_index += 1

    def run(
        self,
        sampler,
        weights: Tuple[float, float, float],
        kappa_theta: torch.Tensor,
        kappa_phi: torch.Tensor,
    ) -> None:
        """
        Generate complete chain by sequentially calling FisherSampler.

        The sequential loop is necessary because each step's mean direction
        depends on curr_bond, which is only known after the previous step.

        Args:
            sampler:     FisherSampler instance
            weights:     Mixture weights (T, Gp, Gn)
            kappa_theta: Polar concentrations,     shape (K,)
            kappa_phi:   Azimuthal concentrations, shape (K,)
        """
        self.init_chains()

        for i in range(self._monomer_index, self.total_length - 1):
            next_bond, tau = sampler(
                weights=weights,
                curr_bond=self.curr_bond,
                prev_bond=self.prev_bond,
                kappa_theta=kappa_theta,
                kappa_phi=kappa_phi,
            )
            self.step(next_bond, tau)