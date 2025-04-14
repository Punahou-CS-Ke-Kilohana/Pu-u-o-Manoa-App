r"""
This file includes the config for the DataLoader of the training script.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import os
import datetime
from easydict import EasyDict as Edict


loader_config = Edict()

loader_config.save_params = {
    'save_root': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'),
    'save_name': f"model_{datetime.datetime.now().year}_{datetime.datetime.now().month}_{datetime.datetime.now().day}"
}

loader_config.dataloader_params = {
    'batch_size': 5,
    'color_channels': 3,
    'classes': 15,
    'initial_dims': (256, 256)
}
