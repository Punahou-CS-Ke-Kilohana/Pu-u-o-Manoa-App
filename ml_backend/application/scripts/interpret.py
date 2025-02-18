r"""
This module consists of the interpreting script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import torch
from typing import Optional


def interpret(
        device: torch.device = torch.device('cpu'),
        name: Optional[str] = None,
        epoch: Optional[int] = None,
        ikwiad: bool = False
) -> None:
    r"""
    Interprets a trained model.

    Args:
        device (torch.device):
            Utilized device.
        name (str):
            Name the loaded model.
        epoch (int):
            Loaded epoch.
        ikwiad (bool, optional):
            "I know what I am doing" (ikwiad).
            If True, removes all warning messages.
            Defaults to False.

    Returns:
        None
    """
    ...