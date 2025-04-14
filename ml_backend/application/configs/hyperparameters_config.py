r"""
This file includes the config for the hyperparameters of the training script.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

from easydict import EasyDict as Edict


hyperparameters_config = Edict()

# loss hyperparameters
hyperparameters_config.loss = Edict()
hyperparameters_config.loss.method = 'CrossEntropyLoss'
hyperparameters_config.loss.hyperparams = {
    'weight': None,
    'size_average': None,
    'ignore_index': -100,
    'reduce': None,
    'reduction': 'mean',
    'label_smoothing': 0.0
}

# optimizer hyperparameters
hyperparameters_config.optimizer = Edict()
hyperparameters_config.optimizer.method = 'Adam'
hyperparameters_config.optimizer.hyperparams = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False,
    'foreach': None,
    'maximize': False,
    'capturable': False,
    'differentiable': False,
    'fused': None
}

# expose to import
__all__ = ['hyperparameters_config']
