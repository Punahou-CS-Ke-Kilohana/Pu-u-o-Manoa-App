# do NOT touch this unless you REALLY know what you're doing

from easydict import EasyDict as Edict

model_config = Edict()

model_config.training_params = {
    'color_channels': 3,
    'classes': 15,
    'initial_dims': (256, 256)
}

model_config.acts = Edict()
model_config.acts.methods = ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax']
model_config.acts.params = [
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
    {'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True, 'padding_mode': 'zeros'}
]
model_config.conv_pool_params = [
    False,
    False,
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
    {'bias': True}
]
