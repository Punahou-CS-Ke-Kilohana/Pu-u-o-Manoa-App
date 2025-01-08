r"""
This module consists of the dataloader for the Pu-u-o-Manoa-App.
It accesses the images folder in the ml_backend directory for its data.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
todo: make sure this file is compatible with the current format for the images folder.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from ..configs.loader_config import loader_config

classes = loader_config.dataloader_params['classes']


def one_hot(y: torch.Tensor) -> torch.Tensor:
    r"""
    Args:
        y (torch.Tensor):
            The outputs.

    Returns:
        torch.Tensor:
            The one hot encoded outputs.

    Raises:
        ValueError: If invalid values were passed for classes.
    """
    if not isinstance(classes, int) or classes < 1:
        raise ValueError("'classes' must be a positive integer")
    one = torch.zeros(classes, dtype=torch.float)
    return one.scatter_(0, torch.tensor(y), value=1)


# load data
data = datasets.ImageFolder(
    root=loader_config.root,
    transform=loader_config.transform,
    target_transform=one_hot
)

# dataloader creation
loader = DataLoader(
    data,
    batch_size=loader_config.dataloader_params['batch_size'],
    shuffle=loader_config.dataloader_params['shuffle'],
    num_workers=loader_config.dataloader_params['num_workers'],
    pin_memory=loader_config.dataloader_params['pin_memory']
)

# expose to import
__all__ = ['loader']
