import numpy as np
from torch.utils.data import Dataset


class randomNormalGen(Dataset):
    def __init__(self, n_particles):
        self.N = 1000
        self.X = 7 + np.random.randn(self.N, n_particles)
        self.Y = 20 + np.random.randn(self.N, 1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
