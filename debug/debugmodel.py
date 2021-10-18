import torch
import torch.nn as nn
from debugdata import randomNormalGen
from torch.utils.data import DataLoader


class referenceNetwork1(nn.Module):
    """
    Network based on the description of the reference paper. Assumptions have been made to determine this
    architecture since an exact description was not given. This Network is the best performing network on the
    Drell-Yan Weights.
    Reference:https://arxiv.org/pdf/2008.10949.pdf
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

    def forward(self, x):
        y = self.relu(self.fc1(x))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = self.relu(self.fc4(y))
        y = self.relu(self.fc5(y))
        y = self.selu(self.fc6(y))
        return y


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_particles = 6
        self.net = referenceNetwork1(self.n_particles)

    def train(self):
        train_data = randomNormalGen(self.n_particles)
        train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
        mse = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters())

        self.net.train()
        for epoch in range(20):
            for i, data in enumerate(train_loader):
                x_train, y_train = data
                y_pred = self.net(x_train.float())
                loss = mse(y_train.float(), y_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch: " + str(epoch) + ", Loss: " + str(loss))


obj = DNN()
obj.train()
