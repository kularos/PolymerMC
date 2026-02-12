"""
Core utility functions for polymer chain transformations.

This module provides fundamental geometric operations used throughout
the simulation, including rotation via Rodrigues' formula and geodesic
alignment of vectors, plus caching infrastructure.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Any


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
            v = torch.tensor([[1.0], [0.0], [0.0]])  # x-axis, shape (3, 1)
            k = torch.tensor([[0.0], [0.0], [1.0]])  # z-axis
            theta = torch.tensor([[np.pi/2]])         # 90 degrees
            rodrigues_rotation(v, k, theta)
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


class GridCache:
    """
    Manages persistent caching of lookup tables with human-readable keys.

    Cache keys are constructed from the parameters that define uniqueness,
    making it easy to identify and manage cached files.

    Example cache filename:
        VonMisesSampler_w(0.6,0.2,0.2)_k(1e-3,1e7,1000)_p(100000).pkl
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for storing cached grids
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_cache_key(self, **params) -> str:
        """
        Generate human-readable cache key from parameters.

        Creates a filename-safe string that clearly shows what parameters
        define this cached grid.

        Args:
            **params: Any parameters that define cache uniqueness
                Common params: class_name, weights, ps_range, ps_res, p_res

        Returns:
            Human-readable cache key string

        Example:
                cache._make_cache_key(
            ...     class_name="VonMisesSampler",
            ...     weights=(0.6, 0.2, 0.2),
            ...     ps_range=(5.0, 9.0),
            ...     ps_res=1000,
            ...     p_res=100000
            ... )
            'VonMisesSampler_w(0.6,0.2,0.2)_pS(5.0,9.0,1000)_p(100000)'
        """
        parts = []

        # Add class name first
        if 'class_name' in params:
            parts.append(str(params['class_name']))

        # Add weights (w)
        if 'weights' in params:
            w = params['weights']
            parts.append(f"w({w[0]},{w[1]},{w[2]})")

        # Add pS range and resolution (entropy parameterization)
        if 'ps_range' in params and 'ps_res' in params:
            ps_min, ps_max = params['ps_range']
            ps_res = params['ps_res']
            parts.append(f"pS({ps_min},{ps_max},{ps_res})")

        # Add probability resolution (p)
        if 'p_res' in params:
            parts.append(f"p({params['p_res']})")

        return "_".join(parts)

    def load(self, **params) -> Optional[torch.Tensor]:
        """
        Load grid from cache if it exists.

        Args:
            **params: Parameters defining the cache key
                Required: class_name
                Typical: weights, ps_range, ps_res, p_res

        Returns:
            Cached grid tensor, or None if not found

        Example:
                grid = cache.load(
            ...     class_name="VonMisesSampler",
            ...     weights=(0.6, 0.2, 0.2),
            ...     ps_range=(5.0, 9.0),
            ...     ps_res=1000,
            ...     p_res=100000
            ... )
        """
        cache_key = self._make_cache_key(**params)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if not cache_path.exists():
            return None

        print(f"📦 Loading cached grid: {cache_key}", end="", flush=True)

        with open(cache_path, 'rb') as f:
            grid_np = pickle.load(f)

        print(" ✓")
        return torch.from_numpy(grid_np).to(torch.float32)

    def save(self, grid: np.ndarray, **params) -> None:
        """
        Save grid to cache.

        Args:
            grid: NumPy array to cache
            **params: Parameters defining the cache key (same as load())

        Example:
                cache.save(
            ...     grid,
            ...     class_name="VonMisesSampler",
            ...     weights=(0.6, 0.2, 0.2),
            ...     ps_range=(5.0, 9.0),
            ...     ps_res=1000,
            ...     p_res=100000
            ... )
        """
        cache_key = self._make_cache_key(**params)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        with open(cache_path, 'wb') as f:
            pickle.dump(grid, f)

        print(f"💾 Cached grid: {cache_key}")

    def clear(self, **params) -> bool:
        """
        Delete specific cache file.

        Args:
            **params: Parameters defining the cache key

        Returns:
            True if file was deleted, False if not found
        """
        cache_key = self._make_cache_key(**params)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if cache_path.exists():
            cache_path.unlink()
            print(f"🗑️  Deleted cache: {cache_key}")
            return True
        return False

    def clear_all(self) -> int:
        """
        Delete all cache files in the cache directory.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1

        if count > 0:
            print(f"🗑️  Deleted {count} cache files")
        return count

    def list_cached(self) -> list[str]:
        """
        List all cached grid keys.

        Returns:
            List of cache key strings (without .pkl extension)
        """
        return [f.stem for f in self.cache_dir.glob("*.pkl")]