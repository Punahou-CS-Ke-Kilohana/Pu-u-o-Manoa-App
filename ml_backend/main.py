r"""
This file is the execution script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

from typing import Optional
import torch

from application.scripts.train import train


def main(name: str, device: Optional[torch.device] = None, *, ikwiad=False) -> int:
    r"""
    The main execution script for the Pu-u-o-Manoa-App.

    Args:
        name (str):
            The method name.
        device (torch.device):
            Utilized device.
        ikwiad (bool, optional):
            "I know what I am doing" (ikwiad).
            If True, removes all warning messages.
            Defaults to False.

    Returns:
        None
    """
    if device is None:
        device = torch.device('cpu')
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
    methods[name](device=device, ikwiad=ikwiad)
    return 0


if __name__ == '__main__':
    import torch.multiprocessing as mp
    from method_selector import method_name

    # device
    loc_device = None
    if torch.backends.mps.is_available():
        loc_device = torch.device('mps')

    # debug issues
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    # main script
    main(name=method_name, device=loc_device)
