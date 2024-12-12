r"""
This file includes the config for the DataLoader of the training script.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import os
import torch
from torchvision import transforms
from easydict import EasyDict as Edict


loader_config = Edict()

loader_config.root = os.path.join(os.path.join((os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'images', 'local'))

loader_config.transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

loader_config.dataloader_params = {
    'classes': 15,
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 0,
    'pin_memory': True,
}

# expose to import
__all__ = ['loader_config']
