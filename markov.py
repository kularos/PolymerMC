import torch
from MCMC.PolymerMC.polymer_sim.core import rodrigues_torch
import numpy as np

class TorsionMCMC:
    def __init__(self, chain_length, n_chains=50, n_batches=1, device="cpu"):
        self._device = device
        self.chain_length = chain_length
        self.n_chains = n_chains
        self.n_batches = n_batches  # New: Number of kappa levels (K)

        #
        self.batch_vec_shape = (3, n_chains, n_batches)
        self.shape = (3, chain_length, n_chains, n_batches)

        self.prev_bond = None
        self.curr_bond = None
        self._chains = None
        self._i = 0

    def init_chains(self, alignment=1 / 3):

        # Set initial bond hybridization angle across all batches
        init_prev = torch.tensor((0.0, (1 - alignment ** 2) ** 0.5, alignment), device=self._device)
        init_curr = torch.tensor((0.0, 0.0, 1.0), device=self._device)

        self.prev_bond = init_prev.view(3, 1, 1).expand(self.batch_vec_shape).to(torch.float64)
        self.curr_bond = init_curr.view(3, 1, 1).expand(self.batch_vec_shape).to(torch.float64)

        # Slicing: [Batch, Chain, Atom, Coord]
        self._chains = torch.zeros(self.shape, device=self._device, dtype=torch.float64)
        self._chains[:, 0, :, :] -= self.prev_bond
        self._chains[:, 2, :, :] += self.curr_bond
        self._i = 1

    def iter_chains(self, tau):
        # Rodrigues now operates on (K, N, 3) vectors and (K, N, 1) angles
        next_bond = rodrigues_torch(self.prev_bond, self.curr_bond, tau)
        self._chains[:, self._i + 1] = self._chains[:, self._i] + next_bond

        self.prev_bond = self.curr_bond
        self.curr_bond = next_bond
        self._i += 1

    def run(self, sample_batch):
        """
        sample_batch shape: (K, N, L, 1) or (K, N, L)
        """
        if sample_batch.dim() == 4:
            sample_batch = sample_batch.squeeze(-1)

        self.init_chains()
        # Iterate through the length of the polymer
        for _ in range(self._i, self.chain_length - 1):
            # tau_i shape: (K, N)
            tau_i = sample_batch[:, self._i]
            self.iter_chains(tau_i)

    @property
    def chains(self):
        return self._chains

