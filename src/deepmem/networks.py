import torch.nn as nn

__all__ = ["referenceNetwork1", "referenceNetwork2", "res2", "resBlock", "resNetwork"]


def __dir__():
    return __all__


class referenceNetwork1(nn.Module):
    """
    Network based on the description of the reference paper. Assumptions have been made to determine this
    architecture since an exact description was not given. This Network is the best performing network on the
    Drell-Yan Weights.
    Reference:https://arxiv.org/pdf/2008.10949.pdf

    6 Fully connected layers
    """

    def __init__(self, n_particles):
        super().__init__()
        self.fc1 = nn.Linear(n_particles, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, n_particles)
        self.fc6 = nn.Linear(n_particles, 1)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, x):
        y = self.relu(self.fc1(x))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = self.relu(self.fc4(y))
        y = self.relu(x + self.fc5(y))
        y = self.selu(self.fc6(y))
        return y


class res2(nn.Module):
    def __init__(self, n_particles):
        super().__init__()
        self.fc1 = nn.Linear(n_particles, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, n_particles)
        self.fc7 = nn.Linear(n_particles, 1)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, x):
        y = self.relu(self.fc1(x))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = self.relu(self.fc4(y))
        y = self.relu(self.fc5(y))
        y = self.relu(x + self.fc6(y))
        y = self.selu(self.fc7(y))
        return y


class referenceNetwork2(nn.Module):
    """
    Same as referenceNetwork1 but with batchnorms on the final 2 layers
    """

    def __init__(self, n_particles):
        super().__init__()
        self.fc1 = nn.Linear(n_particles, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        y = self.relu(self.fc1(x))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = self.relu(self.fc4(y))
        y = self.relu(self.bn1(self.fc5(y)))
        y = self.selu(self.bn2(self.fc6(y)))
        return y


class resBlock(nn.Module):
    def __init__(self, n, n_nodes=128):
        super().__init__()
        self.fc1 = nn.Linear(n, n_nodes)
        self.fc2 = nn.Linear(n_nodes, n)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.bn2 = nn.BatchNorm1d(n)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.fc1(x)))
        y = self.relu(x + self.bn2(self.fc2(y)))
        return y


class resNetwork(nn.Module):
    def __init__(self, n_particles):
        super().__init__()
        self.fc1 = nn.Linear(n_particles, 128)
        self.resblock1 = resBlock(128)
        self.resblock2 = resBlock(128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, x):
        y = self.relu(self.fc1(x))
        y = self.resblock1(y)
        y = self.resblock2(y)
        y = self.relu(self.fc2(y))
        return y
