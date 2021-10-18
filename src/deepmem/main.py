import time  # used only for runtime testing

import torch

from deepmem.data import DataManager
from deepmem.model import DNN
from deepmem.options import parse_args


def main():
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
