import time # used only for runtime testing
from data import CustomDataset, DataManager
from torch.utils.data import Dataset, DataLoader
import sklearn
from options import parse_args
import uproot
import math
import torch
import numpy as np

class datautils:
    def __init__(self, manager, opts, scaler=None):
        self.manager = manager
        self.opts = opts
        self.scaler = scaler

        file = uproot.open(self.manager.args["output_paths"][0])
        self.len = file[self.manager.args["output_tree"]].num_entries # Total number of samples
        file.close()

        self.split = self.manager.args["split"]
        self.training_samples = math.floor(self.len*(self.split[0]/sum(self.split))) # training fraction * total length
        self.validation_samples = math.floor(self.len*(self.split[1]/sum(self.split)))
        self.testing_samples = self.len - self.validation_samples - self.training_samples

    def load_validation_set(self):
        for path in self.manager.args["input_paths"]:
            file = uproot.open(path)
            self.inputs = []

            for iter in range(len(self.manager.args["prefixes"])):
                for i in self.manager.args["prefixes"][iter]:
                    self.inputs += [i+x for x in self.manager.args["suffixes"][iter]]

            self.val_X = file[self.manager.args["input_tree"]].arrays(self.inputs, library="pd", entry_start = self.training_samples, entry_stop = self.training_samples+self.validation_samples).values
            file.close()

        for path in self.manager.args["output_paths"]:
            file = uproot.open(path)
            self.val_Y = torch.from_numpy(file[self.manager.args["output_tree"]].arrays(self.manager.args["weights"], library="pd", entry_start = self.training_samples, entry_stop = self.training_samples+self.validation_samples).values.squeeze())

            file.close()
        self.val_Y = - np.log10(self.val_Y)
        self.preprocess_X(mode = "val")
        self.val_X = torch.from_numpy(self.scaler.transform(self.val_X))

    def load_testing_set(self):
        for path in self.manager.args["input_paths"]:
            file = uproot.open(path)
            self.inputs = []

            for iter in range(len(self.manager.args["prefixes"])):
                for i in self.manager.args["prefixes"][iter]:
                    self.inputs += [i+x for x in self.manager.args["suffixes"][iter]]

            self.test_X = file[self.manager.args["input_tree"]].arrays(self.inputs, library="pd", entry_start = self.training_samples+self.validation_samples).values
            file.close()

        for path in self.manager.args["output_paths"]:
            file = uproot.open(path)
            self.test_Y = torch.from_numpy(file[self.manager.args["output_tree"]].arrays(self.manager.args["weights"], library="pd", entry_start = self.training_samples+self.validation_samples).values.squeeze())

            file.close()
        self.test_Y = - np.log10(self.test_Y)
        self.preprocess_X(mode = "test")
        self.test_X = torch.from_numpy(self.scaler.transform(self.test_X))

    def preprocess_X(self, mode):
        phi_indices = []
        for i in range(len(self.inputs)):
           if(self.inputs[i][-3:] == "Phi"):
               phi_indices.append(i)
        if(len(phi_indices)>1):
            for j in range(1,len(phi_indices)):
                if(mode == "val"):
                    self.val_X[:, phi_indices[j]] -= self.val_X[:, phi_indices[0]]
                elif(mode == "test"):
                    self.test_X[:, phi_indices[j]] -= self.test_X[:, phi_indices[0]]
if __name__ == "__main__":
    start = time.time()
    opts = parse_args()
    data_manager = DataManager()
    dataset = CustomDataset(data_manager, opts)

    utils = datautils(data_manager, opts)
    utils.load_validation_set()
    print(utils.val_X.shape, utils.val_Y.shape)

    print("Script time: ", time .time() - start)
