import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--loader", type=str, default="inbuilt")
    parser.add_argument("--device", type=int, required=True)

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()