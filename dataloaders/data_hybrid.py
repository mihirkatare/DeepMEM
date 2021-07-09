import sys
sys.path.append('./') # append the root directory
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import uproot
import os
import json
import time # testing only
from debug.fastloader import FastTensorDataLoader
import math
from sklearn.utils import shuffle

##
# This file contains the Data Manager and Data Loader modules
##

class DataManager():
    def __init__(self, input_file="./input_files/ignore_inp.json"):
        with open(input_file) as f:
            args = json.load(f)
        self.args = args
        if(args["dir_name_processing"]):
            self.args["particles"] = list(map(self.string_processing, self.args["particles"]))

    def string_processing(self, string):
        return( string + "/fCoordinates/fCoordinates." )

class CustomDSHybrid(Dataset):
    def __init__(self, manager):
        self.manager = manager
        self.bs = self.manager.args["batch_size"]
        self.nec = self.manager.args["chunk_entries"] # number of entries in a chunk
        self.bic = math.ceil(self.nec / self.bs) # Batches in a chunk

        file = uproot.open(self.manager.args["output_paths"][0])
        self.len = file[self.manager.args["tree"]].num_entries # Total number of samples
        file.close()

        self.n_chunks = self.len // self.nec

    def __len__(self):
        return self.n_chunks*self.bic

    def loadChunk(self, n):
        for path in self.manager.args["input_paths"]:
            file = uproot.open(path)
            inputlist = []
            for i in self.manager.args["particles"]:
                inputlist.append( file[self.manager.args["tree"]].arrays([i+x for x in self.manager.args["4v_coords"]], library="pd", entry_start = n*self.nec, entry_stop =(n+1)*self.nec).values)
            self.X_chunk = np.stack(inputlist, axis=-1)
            file.close()
        ## Stacking of multiple paths not implemented yet

        for path in self.manager.args["output_paths"]:
            file = uproot.open(path)
            self.Y_chunk = file[self.manager.args["tree"]].arrays(self.manager.args["weights"], library="pd", entry_start = n*self.nec, entry_stop =(n+1)*self.nec).values
            file.close()
        ## Stacking of multiple paths not implemented yet

        if(self.manager.args["shuffle"]):
            self.X_chunk, self.Y_chunk = shuffle(self.X_chunk, self.Y_chunk)


    def __getitem__(self, idx):
        chunkIdx = idx % self.bic
        if(chunkIdx == 0):
            self.loadChunk(idx // self.nec)

        if(chunkIdx == self.bic-1):
            return self.X_chunk[idx*self.bs:], self.Y_chunk[idx*self.bs:]
        else:
            return self.X_chunk[idx*self.bs:(idx+1)*self.bs], self.Y_chunk[idx*self.bs:(idx+1)*self.bs]


class CustomDSHybridMultithread(Dataset):
    def __init__(self, manager):
        self.manager = manager
        self.bs = self.manager.args["batch_size"]
        self.nec = self.manager.args["chunk_entries"] # number of entries in a chunk
        self.bic = math.ceil(self.nec / self.bs) # Batches in a chunk

        file = uproot.open(self.manager.args["output_paths"][0])
        self.len = file[self.manager.args["tree"]].num_entries # Total number of samples
        file.close()

        self.n_chunks = self.len // self.nec

    def __len__(self):
        return self.n_chunks*self.bic

    def loadChunk(self, n):
        for path in self.manager.args["input_paths"]:
            file = uproot.open(path)
            inputlist = []
            for i in self.manager.args["particles"]:
                inputlist.append( file[self.manager.args["tree"]].arrays([i+x for x in self.manager.args["4v_coords"]], library="pd", entry_start = n*self.nec, entry_stop =(n+1)*self.nec).values)
            self.X_chunk = np.stack(inputlist, axis=-1)
            file.close()
        ## Stacking of multiple paths not implemented yet

        for path in self.manager.args["output_paths"]:
            file = uproot.open(path)
            self.Y_chunk = file[self.manager.args["tree"]].arrays(self.manager.args["weights"], library="pd", entry_start = n*self.nec, entry_stop =(n+1)*self.nec).values
            file.close()
        ## Stacking of multiple paths not implemented yet

        if(self.manager.args["shuffle"]):
            self.X_chunk, self.Y_chunk = shuffle(self.X_chunk, self.Y_chunk)


    def __getitem__(self, idx):
        chunkIdx = idx % self.bic
        if(chunkIdx == 0):
            self.loadChunk(idx // self.nec)

        if(chunkIdx == self.bic-1):
            return self.X_chunk[idx*self.bs:], self.Y_chunk[idx*self.bs:]
        else:
            return self.X_chunk[idx*self.bs:(idx+1)*self.bs], self.Y_chunk[idx*self.bs:(idx+1)*self.bs]

def DataloaderMultithread(dataset, num_workers = 2):
    generator = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 1)


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

    dataset = CustomDSHybrid(data_manager)
    generator = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 0)

    for epoch in range(1):
        for batch_idx, data in enumerate(generator):
            x, y = data
            # print(x.shape, y.shape, " -> batch ", batch_idx)

    print("Script time: ", time .time() - start)
