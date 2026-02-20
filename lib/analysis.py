"""
Analysis tools for polymer chain ensembles.

This module provides the PolymerAnalyzer class for computing geometric
and statistical properties of polymer chain ensembles, including:
- End-to-end distances
- Radius of gyration
- Chain alignment and orientation
- Neighbor distributions

The analyzer stores the full raw chain and bisects lazily via
generate_assets(pGamma, pS_idx), allowing the same sampled data to be
viewed at any subdivision level without recomputation.
"""

import torch
import numpy as np
from typing import Tuple
from .core import rodrigues_rotation, align_geodesic


class BisectedView:
    """
    A lightweight view of a polymer chain at a specific bisection level.

    Created by PolymerAnalyzer.generate_assets(pGamma, pS_idx).
    Holds reshaped tensors for a single (pGamma, pS) coordinate and
    exposes all analysis properties the plotters require.

    Shape convention: internal tensors are (3, L, N) where:
        L = 2^(pL - pGamma)  — monomers per sub-chain
        N = 2^pGamma          — number of sub-chains

    Monomer scale (pGamma == pL): L=1, N=2^pL.
    Geometric methods (align_ends, orient_principal_axes) return the
    chain unchanged at this scale — monomers are pre-aligned by definition.
    """

    def __init__(
        self,
        chains: torch.Tensor,
        taus: torch.Tensor,
        pL: int,
        pGamma: int,
    ):
        """
        Args:
            chains: Chain coordinates, shape (3, L, N)
            taus:   Torsion angles,     shape (L, N)
            pL:     log2 of total chain length
            pGamma: log2 of number of bisections for this view
        """
        self._chains = chains      # (3, L, N)
        self._taus   = taus        # (L, N)
        self.pL      = pL
        self.pGamma  = pGamma

        _, self.chain_length, self.n_chains = self._chains.shape
        self.shape = self._chains.shape  # (3, L, N)

        # Precompute pairwise distance matrix
        # chains: (3, L, N) → permute to (N, L, 3) for torch.cdist
        # cdist result: (N, L, L) → permute to (L, L, N)
        x = self._chains.permute(2, 1, 0)              # (N, L, 3)
        self._dist = torch.cdist(x, x).permute(1, 2, 0)  # (L, L, N)

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def chains(self) -> np.ndarray:
        """Chain coordinates as NumPy array, shape (3, L, N)."""
        return self._chains.numpy()

    @property
    def taus(self) -> np.ndarray:
        """Torsion angle samples as NumPy array, shape (L, N)."""
        return self._taus.numpy()

    @property
    def torsion(self) -> np.ndarray:
        """
        Cumulative torsion angles (integrated winding), shape (L, N).

        Returns the running sum of torsion angles along each sub-chain.
        """
        return self.taus.cumsum(axis=0)

    @property
    def dist(self) -> np.ndarray:
        """
        Pairwise distance matrix, shape (L, L, N).

        dist[i, j, k] = distance between monomer i and j in sub-chain k.
        """
        return self._dist.numpy()

    @property
    def centered(self) -> torch.Tensor:
        """Chains centered at their center of mass, shape (3, L, N)."""
        return self._chains - self._chains.mean(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def mean_neighbors(self, radius: float, eps: float = 1e-3) -> np.ndarray:
        """
        Mean number of neighbors within given radius per sub-chain.

        Args:
            radius: Cutoff distance for neighbor counting
            eps:    Tolerance for floating point comparisons

        Returns:
            Mean neighbors per monomer for each sub-chain, shape (N,)
        """
        total_neighbors = (self._dist < radius + eps).sum(dim=(0, 1))
        return ((total_neighbors - self.chain_length) / self.chain_length).numpy()

    # ------------------------------------------------------------------
    # Geometric transformations
    # ------------------------------------------------------------------

    def align_ends(self, axis: int = 2) -> torch.Tensor:
        """
        Align all sub-chains so their end-to-end vectors point along axis.

        At the monomer scale (chain_length == 1), monomers are considered
        pre-aligned and the chain is returned unchanged.

        Args:
            axis: Target axis index (0=x, 1=y, 2=z)

        Returns:
            Aligned chains, shape (3, L, N)
        """
        if self.chain_length == 1:
            # Monomers are pre-aligned by definition
            return self._chains

        # Translate: move first monomer of each sub-chain to origin
        first_monomers = self._chains[:, 0, :].unsqueeze(1)   # (3, 1, N)
        centered_chains = self._chains - first_monomers

        # End-to-end vectors (first → last monomer), shape (3, 1, N)
        end_to_end = centered_chains[:, -1, :].unsqueeze(1)

        # Target direction as unit vector along specified axis
        target = torch.zeros_like(end_to_end)
        target[axis] = 1.0

        return align_geodesic(centered_chains, end_to_end, target)

    def orient_principal_axes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align sub-chains along their principal axes of inertia.

        Performs PCA on each sub-chain's gyration tensor, rotating so
        the largest principal axis → x, medium → y, smallest → z.

        At the monomer scale (chain_length == 1), a point mass has no
        principal axes — returns chain unchanged with zero eigenvalues.

        Returns:
            aligned_chains: Rotated chains, shape (3, L, N)
            eigenvalues:    Principal moments, shape (N, 3), sorted descending
        """
        if self.chain_length == 1:
            # A single point has no principal axes; return as-is
            zero_eigs = torch.zeros(self.n_chains, 3)
            return self._chains, zero_eigs

        # Center each sub-chain at its center of mass
        com = self._chains.mean(dim=1, keepdim=True)     # (3, 1, N)
        centered = self._chains - com                     # (3, L, N)

        # Permute to (N, L, 3) for batched matmul
        c = centered.permute(2, 1, 0)                    # (N, L, 3)

        # Gyration tensor S = (1/L) X^T X, shape (N, 3, 3)
        gyration_tensors = torch.matmul(
            c.transpose(1, 2), c                         # (N, 3, L) @ (N, L, 3)
        ) / self.chain_length

        # Eigendecomposition: S = V Λ V^T
        eigenvalues, eigenvectors = torch.linalg.eigh(gyration_tensors)
        # eigenvalues: (N, 3), eigenvectors: (N, 3, 3)

        # Sort eigenvalues descending (largest principal axis first)
        sort_idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = torch.gather(eigenvalues, 1, sort_idx)

        idx_exp = sort_idx.unsqueeze(1).expand(-1, 3, -1)
        eigenvectors = torch.gather(eigenvectors, 2, idx_exp)

        # Rotate sub-chains: (N, L, 3) @ (N, 3, 3) → (N, L, 3)
        aligned = torch.matmul(c, eigenvectors)

        # Back to (3, L, N)
        return aligned.permute(2, 1, 0), eigenvalues


class PolymerAnalyzer:
    """
    Analyzer for polymer chain ensembles with lazy hierarchical bisection.

    Stores the full raw chain tensors (3, 2^pL, K) and (2^pL, K) where K
    is the number of pS values (batch dimension). Bisection is performed
    lazily on demand via generate_assets(pGamma, pS_idx).

    This design means:
    - One PolymerAnalyzer is instantiated once per weight batch
    - The same object serves the entire (pGamma × pS) animation sweep
    - All 2^pL monomer samples are conserved at every bisection level —
      bisection is a pure reshape with no data dropped or duplicated

    Valid pGamma range: 0 .. pL (inclusive)
        pGamma=0:  1 chain  × 2^pL monomers  — the full chain
        pGamma=pL: 2^pL chains × 1 monomer   — the monomer scale

    Attributes:
        pL:           log2 of total chain length
        total_length: 2^pL total monomers in the raw chain
        n_pS:         K, number of pS values in the batch dimension
    """

    def __init__(
        self,
        torsion_samples: torch.Tensor,
        chains: torch.Tensor,
        pL: int,
        device: str = "cpu",
    ):
        """
        Initialize analyzer with full raw chain data.

        The pGamma parameter from the old API is intentionally removed —
        bisection level is now chosen lazily at asset-generation time.

        Args:
            torsion_samples: Torsion angle samples, shape (1, 2^pL, K)
            chains:          Chain coordinates,     shape (3, 2^pL, K)
            pL:              log2 of total chain length
            device:          Device for tensor operations
        """
        self._device = device
        self.pL = pL
        self.total_length = 2 ** pL

        # Normalise torsion_samples → (2^pL, K)
        if torsion_samples.dim() == 3 and torsion_samples.shape[0] == 1:
            torsion_samples = torsion_samples.squeeze(0)   # (2^pL, K)
        elif torsion_samples.dim() == 2:
            pass  # already (2^pL, K)
        else:
            raise ValueError(
                f"Unexpected torsion_samples shape: {torsion_samples.shape}. "
                f"Expected (1, 2^pL, K) or (2^pL, K)."
            )

        # Normalise chains → (3, 2^pL, K)
        if chains.dim() == 2:
            chains = chains.unsqueeze(-1)  # (3, 2^pL) → (3, 2^pL, 1)

        if chains.shape[1] != self.total_length:
            raise ValueError(
                f"chains.shape[1]={chains.shape[1]} does not match "
                f"2^pL={self.total_length}"
            )

        self.n_pS = chains.shape[2]  # K

        # Store raw tensors — these are never modified after construction
        self._raw_taus   = torsion_samples   # (2^pL, K)
        self._raw_chains = chains            # (3, 2^pL, K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_assets(self, pGamma: int, pS_idx: int = 0) -> BisectedView:
        """
        Return a BisectedView for the requested bisection level and pS index.

        This is a pure reshape — no data is dropped or duplicated.
        All 2^pL monomer samples are present in every valid BisectedView.

        Args:
            pGamma: Bisection level in [0, pL].
                    0  → 1 chain of 2^pL monomers     (full chain view)
                    pL → 2^pL chains of 1 monomer each (monomer scale)
            pS_idx: Index into the K pS-value batch dimension [0, n_pS).

        Returns:
            BisectedView with internal shape (3, L, N) where:
                L = 2^(pL - pGamma)  — monomers per sub-chain
                N = 2^pGamma          — number of sub-chains

        Raises:
            ValueError: if pGamma ∉ [0, pL] or pS_idx ∉ [0, n_pS)
        """
        if not (0 <= pGamma <= self.pL):
            raise ValueError(
                f"pGamma={pGamma} out of valid range [0, {self.pL}]. "
                f"Valid values: 0 (full chain) through {self.pL} (monomer scale)."
            )
        if not (0 <= pS_idx < self.n_pS):
            raise ValueError(
                f"pS_idx={pS_idx} out of range [0, {self.n_pS})."
            )

        N = 2 ** pGamma                        # number of sub-chains
        L = 2 ** (self.pL - pGamma)            # monomers per sub-chain

        # Slice the pS batch dimension
        chains_k = self._raw_chains[:, :, pS_idx]   # (3, 2^pL)
        taus_k   = self._raw_taus[:, pS_idx]        # (2^pL,)

        # Bisect via reshape:
        #   (3, 2^pL) → (3, N, L) → permute → (3, L, N)
        #   Treats the long chain as N contiguous segments of length L
        chains_bisected = (
            chains_k.reshape(3, N, L)
                     .permute(0, 2, 1)
                     .contiguous()
        )  # (3, L, N)

        #   (2^pL,) → (N, L) → transpose → (L, N)
        taus_bisected = (
            taus_k.reshape(N, L)
                  .T
                  .contiguous()
        )  # (L, N)

        return BisectedView(chains_bisected, taus_bisected, self.pL, pGamma)

    @property
    def valid_pGamma_range(self) -> range:
        """All valid pGamma values: 0 through pL inclusive."""
        return range(0, self.pL + 1)