import torch
import numpy as np
import random

class Seed:
    MAX_SEED = 1000000
    _instance = None
    _current = None
    _previous = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Seed, cls).__new__(cls)
        return cls._instance

    @property
    def value(self):
        """Reading Seed().value returns the active seed."""
        return self._current

    @value.setter
    def value(self, val: int):
        """Setting Seed().value = x automatically seeds all libraries."""
        self._previous = self._current

        if val == -1:
            val = random.randint(0, self.MAX_SEED)

        self._current = val
        self._update_global_seed()


        print(f"🌱 Global Seed set to: {self._current}")

    def _update_global_seed(self):
        val = self._current
        # Global Seeding Logic
        random.seed(val)
        np.random.seed(val)
        torch.manual_seed(val)
        if torch.cuda.is_available():
            # Seed all GPUs for MCMC parallel chains
            torch.cuda.manual_seed_all(val)

    @property
    def last(self):
        """Access the previous seed for easy callbacks."""
        return self._previous

    def __repr__(self):
        return f"Seed(current={self._current}, previous={self._previous})"

def rodrigues_torch(v, k, theta, *, vector_dim=0):
    # v and k: (3, N, K)
    # theta: (1, N, K)

    # Normalize axis of rotation on dim 0
    k = k / torch.linalg.norm(k, dim=vector_dim, keepdim=True)

    cos_t = torch.cos(theta)  # (1, N, K)
    sin_t = torch.sin(theta)  # (1, N, K)

    # Term 1: (3, N, K) * (1, N, K) -> Broadcasts 1 to 3
    term1 = v * cos_t

    # Term 2: Cross product must happen on dim 0
    term2 = torch.linalg.cross(k, v, dim=vector_dim) * sin_t

    # Term 3: Dot product (sum over dim 0)
    dot = torch.sum(k * v, dim=vector_dim, keepdim=True)
    term3 = k * dot * (1 - cos_t)

    return term1 + term2 + term3

def align_geodesic(v, initial_dir, final_dir, *, vector_dim=0):
    """
    Rotates tensor 'v' such that 'initial_dir' becomes 'final_dir'.
    Implements geodesic rotation logic from analysis.py.
    """
    # 1. Normalize the direction vectors to ensure valid dot products
    A_norm = initial_dir / torch.linalg.norm(initial_dir, dim=vector_dim, keepdim=True)
    B_norm = final_dir / torch.linalg.norm(final_dir, dim=vector_dim, keepdim=True)

    # 2. Calculate rotation axis (K) via cross product
    # K is the vector perpendicular to both A and B
    K = torch.linalg.cross(A_norm, B_norm, dim=vector_dim).expand_as(A_norm)

    # 3. Calculate rotation angle (theta) via dot product
    # Clamp to [-1, 1] to prevent NaN from floating point errors
    cos_theta = torch.sum(A_norm * B_norm, dim=vector_dim, keepdim=True)
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

    # 4. Apply the rotation to the input tensor v
    return rodrigues_torch(v, K, theta, vector_dim=vector_dim)