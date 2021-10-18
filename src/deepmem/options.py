import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--loader", type=str, default="hybridMT")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--inputfile", type=str, default="input_files/input.json")
    parser.add_argument("--mode", type=str, help="'train' or 'test'", default="train")
    return parser.parse_args()
