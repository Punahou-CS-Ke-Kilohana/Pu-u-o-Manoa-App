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

# new
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
import os

classes = loader_config.dataloader_params['classes']

# new 
class RobustImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: Skipping corrupt image {path}: {str(e)}")
            # Recursive call to the next index
            # Be careful with edge cases (index overflow)
            if index + 1 >= len(self):
                index = 0
            return self.__getitem__(index + 1)

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
    
    # Ensure y is within valid range
    if torch.is_tensor(y):
        y_value = y.item()
    else:
        y_value = y
        
    if y_value >= classes:
        raise ValueError(f"Label index {y_value} exceeds number of classes ({classes})")
        
    one = torch.zeros(classes, dtype=torch.float)
    return one.scatter_(0, torch.tensor(y), value=1)

# def one_hot(y: torch.Tensor) -> torch.Tensor:
#     r"""
#     Args:
#         y (torch.Tensor):
#             The outputs.

#     Returns:
#         torch.Tensor:
#             The one hot encoded outputs.

#     Raises:
#         ValueError: If invalid values were passed for classes.
#     """
#     if not isinstance(classes, int) or classes < 1:
#         raise ValueError("'classes' must be a positive integer")
#     one = torch.zeros(classes, dtype=torch.float)
#     return one.scatter_(0, torch.tensor(y), value=1)


# new data loader to deal with corrupt images
data = RobustImageFolder(
    root=loader_config.root,
    transform=loader_config.transform,
    target_transform=one_hot
)

# load data
# data = datasets.ImageFolder(
#     root=loader_config.root,
#     transform=loader_config.transform,
#     target_transform=one_hot
# )

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
