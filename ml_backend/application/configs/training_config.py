r"""
This file includes the config for the training elements of the training script.
This config doesn't control the core elements of the training architecture itself, and is
therefore much less finicky than the other configs.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import os
import datetime
from easydict import EasyDict as Edict


training_config = Edict()

# load path
training_config.loaded_model = None

# iteration saving settings
default_name = f"model_{datetime.datetime.now().year}_{datetime.datetime.now().month}_{datetime.datetime.now().day}"
training_config.save_params = {
    'save_root': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'),
    'save_name': f"model_{datetime.datetime.now().year}_{datetime.datetime.now().month}_{datetime.datetime.now().day}"
}
training_config.save_gap = 2

# epochs
training_config.epochs = 3

# expose to import
__all__ = ['training_config']
