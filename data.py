import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import uproot
from pathlib import Path
import json
import time # used only for runtime testing
from options import parse_args
import math
import torch
##
# This file contains the Data Manager and Data Loader modules
##

class DataManager():
    def __init__(self, input_file="input_files/input.json"):
        # input_file argument should be given the .json input file to be parsed

        with open(input_file) as f:
            args = json.load(f)
        self.args = args
        if(args["dir_name_processing"]):
            self.args["particles"] = list(map(self.string_processing, self.args["particles"]))

        # Resolve relative paths to absolute paths
        self.args["input_paths"] = list(map( (lambda x : Path(x).resolve()), self.args["input_paths"]))
        self.args["output_paths"] = list(map( (lambda x : Path(x).resolve()), self.args["output_paths"]))

    def string_processing(self, string):
        return( string + "/fCoordinates/fCoordinates." )

class CustomDataset(Dataset):
    def __init__(self, manager, opts, whichChunk=None):
        self.manager = manager
        self.opts = opts

        file = uproot.open(self.manager.args["output_paths"][0])
        self.len = file[self.manager.args["output_tree"]].num_entries # Total number of samples
        file.close()

        if(self.opts.loader == "hybrid" or self.opts.loader == "hybridMT"):
            self.bs = self.manager.args["batch_size"]
            self.nec = self.manager.args["chunk_entries"] # number of entries in a chunk
            self.bic = math.ceil(self.nec / self.bs) # Batches in a chunk
            self.n_chunks = self.len // self.nec
            if(self.opts.loader == "hybridMT"):
                self.whichChunk = whichChunk # specifies which chunk to load

    def __len__(self):
        if(self.opts.loader == "inbuilt"):
            return self.len
        elif(self.opts.loader == "hybrid"):
            return self.n_chunks*self.bic
        elif(self.opts.loader == "hybridMT"):
            return self.bic

    def load_chunk(self, idx=None, n=None):
        for path in self.manager.args["input_paths"]:
            file = uproot.open(path)
            inputs = []

            for iter in range(len(self.manager.args["prefixes"])):
                for i in self.manager.args["prefixes"][iter]:
                    inputs += [i+x for x in self.manager.args["suffixes"][iter]]

            if(self.opts.loader == "inbuilt"):
                X = file[self.manager.args["input_tree"]].arrays(inputs, library="pd", entry_start =idx, entry_stop =idx+1).values

            elif(self.opts.loader == "hybrid"):
                self.X_chunk = file[self.manager.args["input_tree"]].arrays(inputs, library="pd", entry_start = n*self.nec, entry_stop =(n+1)*self.nec).values

            elif(self.opts.loader == "hybridMT"):
                self.X_chunk = file[self.manager.args["input_tree"]].arrays(inputs, library="pd", entry_start = self.whichChunk*self.nec, entry_stop =(self.whichChunk+1)*self.nec).values

            # self.X_chunk = np.stack(inputlist, axis=-1)
            file.close()
        ## Stacking of multiple paths not implemented yet

        for path in self.manager.args["output_paths"]:
            file = uproot.open(path)

            if(self.opts.loader == "inbuilt"):
                Y = file[self.manager.args["output_tree"]].arrays(self.manager.args["weights"], library="pd", entry_start =idx, entry_stop =idx+1).astype("float32").values
            elif(self.opts.loader == "hybrid"):
                self.Y_chunk = file[self.manager.args["output_tree"]].arrays(self.manager.args["weights"], library="pd", entry_start = n*self.nec, entry_stop =(n+1)*self.nec).values
            elif(self.opts.loader == "hybridMT"):
                self.Y_chunk = file[self.manager.args["output_tree"]].arrays(self.manager.args["weights"], library="pd", entry_start = self.whichChunk*self.nec, entry_stop =(self.whichChunk+1)*self.nec).values

            file.close()
        ## Stacking of multiple paths not implemented yet
        if(self.opts.loader == "inbuilt"):
            return X.squeeze(), Y.squeeze()
    def __getitem__(self, idx):
        if(self.opts.loader == "inbuilt"):
            return self.load_chunk(idx=idx)

        elif(self.opts.loader == "hybrid"):
            chunkIdx = idx % self.bic
            if(chunkIdx == 0):
                self.load_chunk(n = idx // self.nec)

            if(chunkIdx == self.bic-1):
                return self.X_chunk[chunkIdx*self.bs:], self.Y_chunk[chunkIdx*self.bs:]
            else:
                return self.X_chunk[chunkIdx*self.bs:(chunkIdx+1)*self.bs], self.Y_chunk[chunkIdx*self.bs:(chunkIdx+1)*self.bs]

        elif(self.opts.loader == "hybridMT"):
            self.load_chunk()
            if(idx == self.bic-1):
                return self.X_chunk[idx*self.bs:], self.Y_chunk[idx*self.bs:]
            else:
                return self.X_chunk[idx*self.bs:(idx+1)*self.bs], self.Y_chunk[idx*self.bs:(idx+1)*self.bs]

if __name__ == "__main__":
    '''
    This script is for testing the runtime of the dataloader.
    It has a similar structure to a common PyTorch training loop.
    '''
    start = time.time()
    opts = parse_args()
    data_manager = DataManager()
    dataset = CustomDataset(data_manager, opts)
    generator = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 0)

    for batch_idx, data in enumerate(generator):
        x, y = data
        print(x.shape, y.shape, " -> batch ", batch_idx)
    print("Script time: ", time .time() - start)
