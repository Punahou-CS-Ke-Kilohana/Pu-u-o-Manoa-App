r"""
This file includes the config for the training elements of the training script.
This config doesn't control the core elements of the training architecture itself, and is
therefore much less finicky than the other configs.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

from easydict import EasyDict as Edict


training_config = Edict()

training_config.epochs = 1_000
