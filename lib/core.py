"""
Core utility functions for polymer chain transformations.

This module provides fundamental geometric operations used throughout
the simulation, including rotation via Rodrigues' formula and geodesic
alignment of vectors.
"""
import os
import hashlib
from pathlib import Path
import pickle

import torch
import numpy as np
from typing import Optional, Tuple




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


def rodrigues_rotation(
        v: torch.Tensor,
        k: torch.Tensor,
        theta: torch.Tensor,
        vector_dim: int = 0
) -> torch.Tensor:
    """
    Rotate vectors using Rodrigues' rotation formula.

    Implements the rotation of vector v around axis k by angle theta:
        v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1 - cos(θ))

    This is more numerically stable than constructing rotation matrices
    and works efficiently with batched tensors.

    Args:
        v: Vectors to rotate, shape (..., 3, ...)
        k: Rotation axes (will be normalized), shape matching v
        theta: Rotation angles in radians, broadcastable to v
        vector_dim: Dimension along which vector components (x,y,z) lie

    Returns:
        Rotated vectors, same shape as v

    Example:
        tensor([[0.0], [1.0], [0.0]])  # rotated to y-axis
    """
    # Normalize rotation axis
    k_norm = k / torch.linalg.norm(k, dim=vector_dim, keepdim=True)

    # Precompute trig functions
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Rodrigues formula: three terms
    # Term 1: Component parallel to rotation
    term1 = v * cos_theta

    # Term 2: Component perpendicular to k (cross product)
    term2 = torch.linalg.cross(k_norm, v, dim=vector_dim) * sin_theta

    # Term 3: Component parallel to k (dot product)
    dot_product = torch.sum(k_norm * v, dim=vector_dim, keepdim=True)
    term3 = k_norm * dot_product * (1 - cos_theta)

    return term1 + term2 + term3


def align_geodesic(
        v: torch.Tensor,
        initial_dir: torch.Tensor,
        final_dir: torch.Tensor,
        vector_dim: int = 0
) -> torch.Tensor:
    """
    Rotate tensor v to align initial_dir with final_dir via shortest path.

    This performs a geodesic (shortest arc) rotation that maps one
    direction vector onto another, applying the same rotation to all
    components of v.

    The rotation axis is perpendicular to both directions (via cross product),
    and the rotation angle is computed from their dot product.

    Args:
        v: Tensor to rotate
        initial_dir: Current direction vector
        final_dir: Target direction vector
        vector_dim: Dimension along which vector components lie

    Returns:
        Rotated tensor v

    Notes:
        - Handles anti-parallel vectors (180° rotation) via clamping
        - If initial_dir == final_dir, returns v unchanged
    """
    # Normalize direction vectors to ensure valid dot products
    initial_norm = initial_dir / torch.linalg.norm(
        initial_dir, dim=vector_dim, keepdim=True
    )
    final_norm = final_dir / torch.linalg.norm(
        final_dir, dim=vector_dim, keepdim=True
    )

    # Rotation axis: perpendicular to both directions
    rotation_axis = torch.linalg.cross(initial_norm, final_norm, dim=vector_dim)
    rotation_axis = rotation_axis.expand_as(initial_norm)

    # Rotation angle from dot product
    # Clamp to [-1, 1] to prevent NaN from floating point errors
    cos_angle = torch.sum(initial_norm * final_norm, dim=vector_dim, keepdim=True)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

    # Apply rotation
    return rodrigues_rotation(v, rotation_axis, angle, vector_dim=vector_dim)


# Backwards compatibility aliases
rodrigues_torch = rodrigues_rotation  # Keep old name for now
