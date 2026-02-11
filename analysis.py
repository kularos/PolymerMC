import torch
from MCMC.PolymerMC.polymer_sim.core import rodrigues_torch, align_geodesic

class PolymerAnalyzer:
    def __init__(self, sample, chains, device="cpu"):
        self._device = device
        self._taus = sample
        self._chains = chains
        self.shape = self.chains.shape

        x = self._chains.permute(2, 1, 0)
        self._dist = torch.cdist(x, x).permute(1, 2, 0)

        _, self.chain_length, self.n_chains = self.shape

    @property
    def chains(self):
        return self._chains.numpy()

    @property
    def taus(self):
        return self._taus.squeeze().numpy()

    @property
    def torsion(self):
        return self.taus.cumsum(axis=0)

    @property
    def dist(self):
        return self._dist.numpy()

    @property
    def centered(self):
        return self.chains - self.chains.mean(dim=1, keepdim=True)


    def mean_neighbors(self, radius, eps=1E-3):
        total_neighbors = (self._dist < radius + eps).sum(dim=(0, 1))
        return (total_neighbors - self.shape[1]) / self.shape[1]

    def align_ends(self, axis=2):
        # 1. Translation: First point to Origin
        # self.chains shape: (n_chains, chain_length, 3)
        first_points = self._chains[:, 0, :].unsqueeze(1)
        centered_chains = self._chains - first_points

        # 2. Get the current end-to-end vectors
        # A is the vector from the first monomer to the last
        A = centered_chains[:, -1, :].unsqueeze(1)

        # 3. Define the target (Z-axis)
        # We create a target vector for every chain in the batch
        B = torch.zeros(A.shape, device=self._device, dtype=torch.float64)
        B[axis] = 1

        aligned_chains = align_geodesic(centered_chains, A, B)

        return aligned_chains

    def orient_mass(self):
        """
        Centers each chain and rotates it so the principal axes
        align with the Cartesian axes (Z, X, Y).
        """
        # 1. Center of Mass (CoM) subtraction using float64 precision
        com = self.chains.mean(dim=1, keepdim=True)
        centered_chains = self.chains - com

        # 2. Compute Gyration Tensor (S)
        # S = (1/N) * (X^T * X)
        n = self.chain_length
        # transpose(1, 2) ensures we get a (3, 3) matrix per chain
        gyration_tensors = torch.matmul(centered_chains.transpose(1, 2), centered_chains) / n

        # 3. Eigen-decomposition
        # L: eigenvalues (3,), V: eigenvectors (3, 3) per batch
        L, V = torch.linalg.eigh(gyration_tensors)

        # 4. Sort in descending order (Largest eigenvalue = Principal axis)
        idx = torch.argsort(L, descending=True)

        # Reorder eigenvalues and eigenvectors across the batch
        # We use gather to maintain the (batch, 3) and (batch, 3, 3) shapes
        L = torch.gather(L, 1, idx)

        # Advanced gathering for the 3D eigenvector tensor
        idx_expanded = idx.unsqueeze(1).expand(-1, 3, -1)
        V = torch.gather(V, 2, idx_expanded)

        # 5. Rotate the chains
        # Use matmul to transform (n_chains, chain_length, 3) by (n_chains, 3, 3)
        aligned_chains = torch.matmul(centered_chains, V)

        return aligned_chains, L

