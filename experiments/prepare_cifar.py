"""Download CIFAR-10 and prepare the fixed training/validation split."""
import argparse
from pathlib import Path
from experiments.vision_data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar10'))
    prepare(parser.parse_args().root)


if __name__ == '__main__':
    main()
