import time  # used only for runtime testing

import hist
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error as mape
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data import CustomDataset, DataManager
from networks import referenceNetwork1
from options import parse_args
from utils import datautils


class DNN:
    def __init__(self, manager, opts):
        self.manager = manager
        self.opts = opts
        self.n_particles = 0
        for i in range(len(self.manager.args["prefixes"])):
            self.n_particles += len(self.manager.args["prefixes"][i]) * len(
                self.manager.args["suffixes"][i]
            )
        self.net = referenceNetwork1(self.n_particles)

    def train(self, device, epochs):
        dataset = CustomDataset(self.manager, self.opts)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        mse = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.manager.args["LearningRate"]
        )
        scheduler = StepLR(
            optimizer, step_size=25, gamma=0.7
        )  # Switched off currently, will be used or hyperparameter tuning

        # load validation set
        self.utils = datautils(self.manager, self.opts, dataset.scaler)
        self.utils.load_validation_set()

        self.net.to(device)
        self.net.train()

        startloop = time.time()
        lowest_loss = float("inf")
        for epoch in range(epochs):
            startepoch = time.time()
            for i, data in enumerate(train_loader):
                # Training
                x_train, y_train = (
                    data[0].cuda(device).squeeze(),
                    data[1].cuda(device).squeeze(),
                )
                y_pred = self.net(x_train.float()).squeeze()
                loss = mse(y_train.float(), y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if loss.detach().item() < lowest_loss:
                lowest_loss = loss.detach().item()
                print("Saving ->")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss,
                    },
                    self.manager.args["save_lowest_loss_model_at"],
                )
            # Validation
            self.net.eval()
            with torch.no_grad():
                val_y_pred = self.net(self.utils.val_X.float().cuda(device)).squeeze()
                val_loss = (
                    mse(self.utils.val_Y.float().cuda(device), val_y_pred)
                    .detach()
                    .item()
                )
            self.net.train()
            scheduler.step()
            print(
                "Epoch: "
                + str(epoch + 1)
                + ", Loss: "
                + str(loss)
                + ", valLoss: "
                + str("%.5f" % val_loss)
                + ", Time: "
                + str("%.1f" % (time.time() - startepoch))
            )
        print("Loop time: ", time.time() - startloop)

        # Save model for later use
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
            },
            "post/finalepoch_tempsave.pth",
        )

    def inference(self, device):
        self.net.to(device)
        self.net.eval()
        saved_dict = torch.load(self.manager.args["load_saved_model_from"])
        self.net.load_state_dict(saved_dict["model_state_dict"])
        scaler = joblib.load(self.manager.args["load_scaler_from"])
        mse = nn.MSELoss()
        # load testing set
        utils = datautils(self.manager, self.opts, scaler)
        utils.load_testing_set()

        with torch.no_grad():
            test_y_pred = self.net(utils.test_X.float().cuda(device)).squeeze()
            test_loss = (
                mse(utils.test_Y.float().cuda(device), test_y_pred).detach().item()
            )
        print(
            "Testing Loss: "
            + str("%.5f" % test_loss)
            + " MAPE: "
            + str(mape(utils.test_Y.float(), test_y_pred.cpu()))
        )
        nbins = 100
        hist_1 = hist.Hist(
            hist.axis.Regular(nbins, 0, 10, name="-log_10(DY weight)")
        ).fill(utils.test_Y.numpy())
        hist_2 = hist.Hist(
            hist.axis.Regular(nbins, 0, 10, name="-log_10(DY weight)")
        ).fill(test_y_pred.detach().cpu().numpy())
        _ = plt.figure(figsize=(10, 8))
        main_ax_artists, sublot_ax_arists = hist_1.plot_ratio(
            hist_2,
            rp_ylabel=r"Ratio",
            rp_num_label="Test Dataset",
            rp_denom_label="DNN Prediction",
            rp_uncert_draw_type="line",
        )
        plt.yticks(np.arange(-5, 6))
        plt.grid()
        plt.savefig(self.manager.args["save_testing_histogram_at"])


if __name__ == "__main__":
    start = time.time()
    opts = parse_args()
    data_manager = DataManager(input_file=opts.inputfile)
    obj = DNN(data_manager, opts)
    if opts.mode == "train":
        obj.train(device=torch.device(opts.device), epochs=opts.epochs)
    if opts.mode == "test":
        obj.inference(device=torch.device(opts.device))
    print("Script time: ", time.time() - start)
