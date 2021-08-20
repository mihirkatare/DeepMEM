import numpy as np
from matplotlib.figure import Figure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import CustomDataset, DataManager
from torch.utils.data import Dataset, DataLoader
from options import parse_args
import time # used only for runtime testing
from torch.optim.lr_scheduler import StepLR
from utils import datautils
import joblib

class referenceNetwork1(nn.Module):
    '''
    Network based on the description of the reference paper. Assumptions have been made to determine this
    architecture since an exact description was not given. This Network is the best performing network on the
    Drell-Yan Weights.
    Reference:https://arxiv.org/pdf/2008.10949.pdf
    '''
    def __init__(self, n_particles):
        super(referenceNetwork1, self).__init__()
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

class DNN():
    def __init__(self, manager, opts):
        self.manager = manager
        self.opts = opts
        self.n_particles = 0
        for i in range(len(self.manager.args["prefixes"])):
            self.n_particles += len(self.manager.args["prefixes"][i]) * len(self.manager.args["suffixes"][i])
        self.net = referenceNetwork1(self.n_particles)

    def train(self, device, epochs):
        dataset = CustomDataset(self.manager, self.opts)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 0)
        mse = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.manager.args["LearningRate"])
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5) # Switched off currently, will be used or hyperparameter tuning
        # load validation set
        self.utils = datautils(self.manager, self.opts, dataset.scaler)
        self.utils.load_validation_set()

        self.net.to(device)
        self.net.train()


        startloop = time.time()
        lowest_loss = float('inf')
        for epoch in range(epochs):
            startepoch = time.time()
            for i, data in enumerate(train_loader):
                # Training
                x_train, y_train = data[0].cuda(device).squeeze(), data[1].cuda(device).squeeze()
                y_pred = self.net(x_train.float()).squeeze()
                loss = mse(y_train.float(), y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if(loss.detach().item()<lowest_loss):
                lowest_loss = loss.detach().item()
                print("Saving ->")
                torch.save({'epoch': epoch, 'model_state_dict': self.net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),'loss': loss}, self.manager.args["save_lowest_loss_model_at"])
            # Validation
            self.net.eval()
            with torch.no_grad():
                val_y_pred = self.net(self.utils.val_X.float().cuda(device)).squeeze()
                val_loss = mse(self.utils.val_Y.float().cuda(device), val_y_pred).detach().item()
            self.net.train()
            scheduler.step()
            print("Epoch: " + str(epoch+1) + ", Loss: " + str(loss) + ", valLoss: " + str("%.5f" % val_loss) + ", Time: "+ str("%.1f" % (time.time() - startepoch) ) )
        print("Loop time: ", time.time() - startloop)

        # Save model for later use
        torch.save({'epoch': epoch, 'model_state_dict': self.net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),'loss': loss}, "post/finalepoch_tempsave.pth")

    def inference(self, device):
        self.net.to(device)
        self.net.eval()
        saved_dict = torch.load(self.manager.args["load_saved_model_from"])
        self.net.load_state_dict(saved_dict['model_state_dict'])
        scaler = joblib.load(self.manager.args["load_scaler_from"])
        mse = nn.MSELoss()
        # load testing set
        utils = datautils(self.manager, self.opts, scaler)
        utils.load_testing_set()

        with torch.no_grad():
            test_y_pred = self.net(utils.test_X.float().cuda(device)).squeeze()
            test_loss = mse(utils.test_Y.float().cuda(device), test_y_pred).detach().item()
        print("Testing Loss: " + str("%.5f" % test_loss))
        nbins = 100

        fig = Figure()
        ax = fig.subplots()
        ax.hist(utils.test_Y.numpy(), bins=nbins, histtype = "step", color = "r", label = "Test Dataset")
        ax.hist(test_y_pred.detach().cpu().numpy(), bins=nbins, histtype = "step", label = "DNN Prediction")
        ax.set_xlabel(r"$-\log_{10}\,($Drell-Yan MoMEMta Weights$)$")
        ax.set_ylabel("events")
        ax.legend(loc="best")
        # fig.savefig(self.manager.args["save_testing_histogram_at"])
        file_extension = ["png", "pdf"]
        for extension in file_extension:
            fig.savefig(f"post/histogram_{ax.get_yscale()}.{extension}")

        ax.set_yscale("log")
        for extension in file_extension:
            fig.savefig(f"post/histogram_{ax.get_yscale()}.{extension}")

if __name__ == "__main__":
    start = time.time()
    opts = parse_args()
    data_manager = DataManager(input_file = opts.inputfile)
    obj = DNN(data_manager, opts)
    if opts.mode == "train":
        obj.train(device = torch.device(opts.device), epochs = opts.epochs)
    if opts.mode == "test":
        obj.inference(device = torch.device(opts.device))
    print("Script time: ", time.time() - start)
