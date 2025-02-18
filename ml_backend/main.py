r"""
This file is the execution script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import argparse
import torch

from application.scripts.train import train
from application.scripts.interpret import interpret
from application.scripts.validate import validate


def parse():
    r"""
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Main argument parser.")

    # arguments
    parser.add_argument("--script", "-s", type=str, required=True, help="Executed script.")
    parser.add_argument("--device", "-d", type=str, required=False, help="PyTorch device.")
    parser.add_argument("--name", "-n", type=str, required=False, help="Loaded model.")
    parser.add_argument("--epoch", "-e", type=str, required=False, help="Loaded epoch.")
    parser.add_argument("--ikwiad", action="store_true", help="'I know what I am doing.'")
    return parser.parse_args()


def main() -> int:
    r"""
    The main execution script for the Pu-u-o-Manoa-App.

    Returns:
        int: 0.
    """
    # get args
    args = parse()
    # get device
    if args.device == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Utilizing Metal.")
        else:
            device = torch.device('cpu')
            print("Failed utilizing Metal.")
    elif torch.cuda.is_available() and args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Utilizing CUDA.")
        else:
            device = torch.device('cpu')
            print("Failed utilizing CUDA.")
    else:
        device = torch.device('cpu')
        print("Utilizing CPU.")

    # method reference
    methods = {
        'train': train,
        'validate': validate,
        'interpret': interpret
    }
    method_titles = {
        'train': 'Training',
        'validate': 'Validating',
        'interpret': 'Interpreting'
    }
    if args.script not in methods.keys():
        # invalid method
        raise ValueError(
            "Invalid referenced script\n"
            f"({list(methods.keys())})"
        )

    # method execution
    print(f"{method_titles[args.script]}")
    try:
        epoch = int(args.epoch)
    except TypeError or ValueError:
        epoch = args.epoch
    methods[args.script](device=device, name=(args.name or None), epoch=(epoch or None), ikwiad=bool(args.ikwiad))
    return 0


if __name__ == '__main__':
    import torch.multiprocessing as mp

    # debug issues
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    # main script
    main()
