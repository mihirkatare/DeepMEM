import time  # used only for runtime testing

import torch

from data import DataManager
from model import DNN
from options import parse_args

if __name__ == "__main__":
    start = time.time()
    opts = parse_args()
    if opts.device is None:
        opts.device = "cpu"

    data_manager = DataManager(input_file=opts.inputfile)
    obj = DNN(data_manager, opts)
    if opts.mode == "train":
        obj.train(device=torch.device(opts.device), epochs=opts.epochs)
    if opts.mode == "test":
        obj.inference(device=torch.device(opts.device))
    print("Script time: ", time.time() - start)
