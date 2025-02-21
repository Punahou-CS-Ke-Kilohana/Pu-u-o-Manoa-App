r"""
This file includes the config for the DataLoader of the training script.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import os
import torch
from torchvision import transforms
from easydict import EasyDict as Edict


h, w = 256, 256

loader_config = Edict()

# image location path
loader_config.root = os.path.join(os.path.join((os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'images', 'local'))

# image transformation
loader_config.transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

loader_config.interpret_params = {
    'color_channels': 3,
    'initial_dims': (h, w)
}

# input image path
loader_config.image_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'test_images')

# dataloader parameters
loader_config.dataloader_params = {
    'classes': 15,
    'batch_size': 1,
    'shuffle': True,
    'num_workers': 0,
    'pin_memory': True,
}

# expose to import
__all__ = ['loader_config']
