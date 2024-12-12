r"""
This file includes the config for the CNN model of the training script.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

from easydict import EasyDict as Edict


model_config = Edict()

model_config.acts = Edict()
model_config.acts.methods = ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax']
model_config.acts.params = [
    {'inplace': False},
    {'inplace': False},
    {'inplace': False},
    {'inplace': False},
    {'inplace': False},
    {'inplace': False},
    {'inplace': False},
    {'inplace': False},
    {'inplace': False},
    {'dim': None}
]

model_config.conv = Edict()
model_config.conv.sizes = [16, 32, 64, 128]
model_config.conv.conv_params = [
    {'kernel_size': 5, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True, 'padding_mode': 'zeros'},
    {'kernel_size': 5, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True, 'padding_mode': 'zeros'},
    {'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True, 'padding_mode': 'zeros'},
    {'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True, 'padding_mode': 'zeros'},
    {'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True, 'padding_mode': 'zeros'}
]
model_config.conv.pool_params = [
    None,
    None,
    {'kernel_size': 3, 'stride': None, 'padding': 0, 'dilation': 1, 'return_indices': False, 'ceil_mode': False},
    {'kernel_size': 3, 'stride': None, 'padding': 0, 'dilation': 1, 'return_indices': False, 'ceil_mode': False},
    {'kernel_size': 3, 'stride': None, 'padding': 0, 'dilation': 1, 'return_indices': False, 'ceil_mode': False}
]

model_config.dense = Edict()
model_config.dense.sizes = [256, 128, 64, 32]
model_config.dense.params = [
    {'bias': True},
    {'bias': True},
    {'bias': True},
    {'bias': True},
    {'bias': True},
    {'bias': True}
]

# expose to import
__all__ = ['model_config']
