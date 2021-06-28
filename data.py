import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import uproot
import os
import json
import time # testing only
from debug.fastloader import FastTensorDataLoader

##
# This file contains the Data Manager and Data Loader modules
##

class DataManager():
    def __init__(self, input_file="input_files/input.json"):
        with open(input_file) as f:
            args = json.load(f)
        self.args = args
        if(args["dir_name_processing"]):
            self.args["particles"] = list(map(self.string_processing, self.args["particles"]))
    
    def string_processing(self, string):
        return( string + "/fCoordinates/fCoordinates." )
    
class CustomDataset(Dataset):
    def __init__(self, manager):
        self.manager = manager

    def __len__(self):
        file = uproot.open(self.manager.args["output_paths"][0])
        len = file[self.manager.args["tree"]].arrays(self.manager.args["weights"], library="pd").shape[0]
        file.close()
        return len
    
    def getitemhelper(self, idx):
        for path in self.manager.args["input_paths"]:
            file = uproot.open(path)
            inputlist = []
            for i in self.manager.args["particles"]:
                inputlist.append( file[self.manager.args["tree"]].arrays([i+x for x in self.manager.args["4v_coords"]], library="pd", entry_start =idx, entry_stop =idx+1).values)
            X = np.stack(inputlist, axis=-1)
            file.close()
        ## Stacking of multiple paths not implemented yet

        for path in self.manager.args["output_paths"]:
            file = uproot.open(path)
            Y = file[self.manager.args["tree"]].arrays(self.manager.args["weights"], library="pd", entry_start =idx, entry_stop =idx+1).values
            file.close()
        ## Stacking of multiple paths not implemented yet

        return X.squeeze(), Y.squeeze()

    def getall(self):
        for path in self.manager.args["input_paths"]:
            file = uproot.open(path)
            inputlist = []
            for i in self.manager.args["particles"]:
                inputlist.append( file[self.manager.args["tree"]].arrays([i+x for x in self.manager.args["4v_coords"]], library="pd").values)
            X = np.stack(inputlist, axis=-1)
            file.close()
        ## Stacking of multiple paths not implemented yet

        for path in self.manager.args["output_paths"]:
            file = uproot.open(path)
            Y = file[self.manager.args["tree"]].arrays(self.manager.args["weights"], library="pd").values
            file.close()
        ## Stacking of multiple paths not implemented yet

        return X, Y

    def __getitem__(self, idx):
        return self.getitemhelper(idx)
        

# path = "/data/deepmem_debug/tt_20evt.root"
# path_weights = "/data/deepmem_debug/tt_20evt_weights.root"

# file = uproot.open(path_weights)
# a1 = file["t"].arrays(["bb_p4/fCoordinates/fCoordinates.fPt", "bb_p4/fCoordinates/fCoordinates.fEta", "bb_p4/fCoordinates/fCoordinates.fPhi", "bb_p4/fCoordinates/fCoordinates.fM"], library="pd").values.T
# print(a1)
# print(file["t"].show())

# file.close()

if(__name__ == "__main__"):
    start = time.time()
    data_manager = DataManager()
    dataset = CustomDataset(data_manager)
    # generator = DataLoader(dataset, batch_size=2, shuffle=True, num_workers = 3)

    X, Y = dataset.getall()
    generator = FastTensorDataLoader(X,Y, batch_size=2, shuffle=True)
    for batch_idx, data in enumerate(generator):
        x, y = data
        print(x.shape, y.shape, " -> batch ", batch_idx)
    print("Script time: ", time .time() - start)
