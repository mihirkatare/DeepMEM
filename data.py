import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import uproot
import os
import json

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
    
class DataGenerator(Dataset):
    def __init__(self, manager):
        self.manager = manager
    
    def __getitem__(self, idx):
        pass
        

path = "/data/deepmem_debug/tt_20evt.root"
path_weights = "/data/deepmem_debug/tt_20evt_weights.root"

file = uproot.open(path)
a1 = file["t"].arrays(["bb_p4/fCoordinates/fCoordinates.fPt", "bb_p4/fCoordinates/fCoordinates.fEta", "bb_p4/fCoordinates/fCoordinates.fPhi", "bb_p4/fCoordinates/fCoordinates.fM"], library="pd").values.T
print(a1)
# print(file["t"]["weight_TT_time"].array(library="np"))

file.close()

if(__name__ == "__main__"):
    data_manager = DataManager()

