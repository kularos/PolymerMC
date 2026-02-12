"""
Analysis tools for polymer chain ensembles.

This module provides the PolymerAnalyzer class for computing geometric
and statistical properties of polymer chain ensembles, including:
- End-to-end distances
- Radius of gyration
- Chain alignment and orientation
- Neighbor distributions
"""

import torch
import numpy as np
from typing import Tuple

from .core import rodrigues_rotation, align_geodesic


class PolymerAnalyzer:
    """
    Analyzer for polymer chain ensemble properties.

    This class wraps a set of polymer chains and provides methods for
    computing various geometric and statistical properties, including
    distances, alignments, and orientations.

    Attributes:
        shape: (n_chains, chain_length, 3) shape of chain ensemble
        chain_length: Number of monomers per chain
        n_chains: Number of chains in ensemble
    """

    def __init__(
            self,
            torsion_samples: torch.Tensor,
            chains: torch.Tensor,
            device: str = "cpu"
    ):
        """
        Initialize analyzer with chain configurations.

        Args:
            torsion_samples: Torsion angle samples, shape (1, L, N, 1)
            chains: Chain coordinates, shape (3, L, N)
            device: Device for tensor operations
        """
        self._device = device
        self._taus = torsion_samples
        self._chains = chains
        self.shape = chains.shape

        # Precompute pairwise distance matrix for efficiency
        # Permute to (N, L, 3) for cdist, then back to (L, N, N)
        x = self._chains.permute(2, 1, 0)
        self._dist = torch.cdist(x, x).permute(1, 2, 0)

        _, self.chain_length, self.n_chains = self.shape

    @property
    def chains(self) -> np.ndarray:
        """Get chain coordinates as NumPy array, shape (3, L, N)."""
        return self._chains.numpy()

    @property
    def taus(self) -> np.ndarray:
        """Get torsion angle samples as NumPy array."""
        return self._taus.squeeze().numpy()

    @property
    def torsion(self) -> np.ndarray:
        """
        Get cumulative torsion angles (integrated winding).

        Returns:
            Cumulative sum of torsion angles, shape (L, N)
        """
        return self.taus.cumsum(axis=0)

    @property
    def dist(self) -> np.ndarray:
        """
        Get pairwise distance matrix.

        Returns:
            Distance matrix, shape (L, N, N)
            dist[i, j, k] = distance between monomer i and j in chain k
        """
        return self._dist.numpy()

    @property
    def centered(self) -> torch.Tensor:
        """Get chains centered at their center of mass."""
        return self._chains - self._chains.mean(dim=1, keepdim=True)

    def mean_neighbors(self, radius: float, eps: float = 1e-3) -> np.ndarray:
        """
        Compute mean number of neighbors within given radius.

        For each chain, counts how many monomers are within the specified
        radius of each other, averaged over all monomers.

        Args:
            radius: Cutoff distance for neighbor counting
            eps: Small epsilon to handle floating point comparisons

        Returns:
            Mean neighbors per monomer for each chain, shape (N,)
        """
        # Count pairs within radius (subtract diagonal)
        total_neighbors = (self._dist < radius + eps).sum(dim=(0, 1))
        return ((total_neighbors - self.shape[1]) / self.shape[1]).numpy()

    def align_ends(self, axis: int = 2) -> torch.Tensor:
        """
        Align all chains so their end-to-end vectors point along specified axis.

        This performs:
        1. Translation: Move first monomer to origin
        2. Rotation: Align end-to-end vector to target axis

        Args:
            axis: Target axis index (0=x, 1=y, 2=z)

        Returns:
            Aligned chains, shape (3, L, N)
        """
        # Translate: first monomer to origin
        first_monomers = self._chains[:, 0, :].unsqueeze(1)
        centered_chains = self._chains - first_monomers

        # Get end-to-end vectors (from first to last monomer)
        end_to_end = centered_chains[:, -1, :].unsqueeze(1)

        # Define target direction (unit vector along specified axis)
        target = torch.zeros_like(end_to_end)
        target[axis] = 1.0

        # Perform geodesic rotation
        return align_geodesic(centered_chains, end_to_end, target)

    def orient_principal_axes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align chains along their principal axes of inertia.

        This performs principal component analysis on each chain's
        gyration tensor, rotating chains so their principal axes
        align with the Cartesian axes (largest → x, medium → y, smallest → z).

        Returns:
            aligned_chains: Rotated chains, shape (3, L, N)
            eigenvalues: Principal moments, shape (N, 3), sorted descending

        Notes:
            The gyration tensor S = (1/N) X^T X gives the second moments
            of the mass distribution. Its eigenvectors define the principal
            axes of the polymer coil.
        """
        # Center each chain at its center of mass
        com = self._chains.mean(dim=1, keepdim=True)
        centered = self._chains - com

        # Compute gyration tensor S = (1/N) X^T X
        # Shape: (N, 3, 3) batch of 3x3 symmetric matrices
        n = self.chain_length
        gyration_tensors = torch.matmul(
            centered.transpose(1, 2),
            centered
        ) / n

        # Eigendecomposition: S = V Λ V^T
        # eigenvalues: (N, 3), eigenvectors: (N, 3, 3)
        eigenvalues, eigenvectors = torch.linalg.eigh(gyration_tensors)

        # Sort eigenvalues in descending order (largest first)
        sort_indices = torch.argsort(eigenvalues, descending=True)

        # Reorder eigenvalues
        eigenvalues = torch.gather(eigenvalues, 1, sort_indices)

        # Reorder eigenvectors (need to gather along dim=2)
        idx_expanded = sort_indices.unsqueeze(1).expand(-1, 3, -1)
        eigenvectors = torch.gather(eigenvectors, 2, idx_expanded)

        # Rotate chains to align with principal axes
        # matmul: (N, L, 3) @ (N, 3, 3) -> (N, L, 3)
        aligned = torch.matmul(centered, eigenvectors)

        # Transpose back to (3, L, N) format
        return aligned.permute(2, 1, 0), eigenvalues
