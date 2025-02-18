r"""
This file is the execution script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import argparse
import torch

from application.scripts.train import train


def parse():
    r"""
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Main argument parser.")

    # arguments
    parser.add_argument("--device", type=str, required=False, help="PyTorch device.")
    parser.add_argument("--ikwiad", action="store_true", help="'I know what I am doing'")
    return parser.parse_args()


def main(name: str) -> int:
    r"""
    The main execution script for the Pu-u-o-Manoa-App.

    Args:
        name (str):
            The method name.

    Returns:
        None
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
        'validate': ...,
        'interpret': ...
    }
    method_titles = {
        'train': 'Training',
        'validate': 'Validating',
        'interpret': 'Interpreting'
    }
    if name not in methods.keys():
        # invalid method
        raise ValueError(
            "'name' must refer to a valid method\n"
            f"({list(methods.keys())})"
        )

    # method execution
    print(f"{method_titles[name]}")
    methods[name](device=device, ikwiad=bool(args.ikwiad))
    return 0


if __name__ == '__main__':
    import torch.multiprocessing as mp
    from method_selector import method_name

    # debug issues
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    # main script
    main(name=method_name)
