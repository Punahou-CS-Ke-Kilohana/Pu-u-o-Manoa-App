import types

import torch
import torch.functional as f
import torch.optim as optim
import torch.nn as nn

from .utils import ParamChecker
#test commit

class TorchCNN(torch.nn.Module):
    def __init__(self, *, status_bars: bool = False, ikwiad: bool = False):
        super(TorchCNN, self).__init__()

        # internal visual settings
        self._ikwiad = ikwiad
        self._status = status_bars

        # network allowed settings
        self.allowed_acts = [
            ''
        ]
        self.allowed_optims = [
            ''
        ]

        # network features
        self._conv_sizes = None
        self._dense_sizes = None
        self._feature_space = None
        self._classifications = None

        # technical network elements
        self._acts = None
        self._optim = None
        self._loss = None
        self._conv = None
        self._pool = None
        self._dense = None

    def set_sizes(self, *, conv_channels: list = None, dense_sizes: list = None) -> None:
        if conv_channels is None:
            conv_channels = [128, 64, 32]
        if dense_sizes is None:
            dense_sizes = [256, 128, 64, 32]
        self._conv_sizes = ['batching', *conv_channels]
        self._dense_sizes = ['flattened', *dense_sizes, 'classes']

    def set_conv(self, *, parameters: list) -> None:
        # check if parameters are formatted correctly
        if not (isinstance(parameters, list) or len(parameters) != len(self._conv_sizes) - 1):
            raise ValueError(f"Convolutional parameters were not formatted correctly: {parameters}")

        # instantiate parameter checker
        conv_checker = ParamChecker(name='conv', ikwiad=self._ikwiad)
        conv_checker.set_types(
            default={
                'kernel_size': 3,
                'stride': 1,
                'padding': 0,
                'dilation': 1,
                'groups': 1,
                'bias': True,
                'padding_mode': 'zeros'
            },
            dtypes={
                'kernel_size': (int, tuple),
                'stride': (int, tuple),
                'padding': (int, tuple),
                'dilation': (int, tuple),
                'groups': int,
                'bias': (bool, int),
                'padding_mode': str
            },
            vtypes={
                'kernel_size': lambda x: (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'stride': lambda x: (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'padding': lambda x: (isinstance(x, int) and 0 <= x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 <= i for i in x)),
                'dilation': lambda x: (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'groups': lambda x: 0 < x,
                'bias': lambda x: True,
                'padding_mode': lambda x: x in ['zeros', 'reflect', 'replicate', 'circular']
            },
            ctypes={
                'kernel_size': lambda x: x,
                'stride': lambda x: x,
                'padding': lambda x: x,
                'dilation': lambda x: x,
                'groups': lambda x: x,
                'bias': lambda x: bool(x),
                'padding_mode': lambda x: x
            }
        )

        # get each layer's parameters
        conv_params = []
        for prm in parameters:
            conv_params.append(conv_checker.check_params(prm))

        # set convolutional layers
        self._conv = []
        for prms in range(len(conv_params)):
            self._conv.append(torch.nn.Conv2d(*self._dense_sizes[prms:prms + 1], **conv_params[prms]))

    def set_dense(self, *, parameters=None):
        # check if parameters are formatted correctly
        if not (isinstance(parameters, list) or len(parameters) < 1):
            raise ValueError(f"Dense parameters were not formatted correctly: {parameters}")

        # instantiate parameter checker
        dense_checker = ParamChecker(name='dense', ikwiad=self._ikwiad)
        dense_checker.set_types(
            default={
                'in_features': None,
                'out_features': None,
                'bias': True
            },
            dtypes={
                'in_features': int,
                'out_features': int,
                'bias': (bool, int)
            },
            vtypes={
                'in_features': lambda x: 0 < x,
                'out_features': lambda x: 0 < x,
                'bias': lambda x: True
            },
            ctypes={
                'in_features': lambda x: x,
                'out_features': lambda x: x,
                'bias': lambda x: bool(x)
            }
        )

        # get each layer's parameters
        dense_params = []
        for prm in parameters:
            dense_params.append(dense_checker.check_params(prm))

        # set dense layers
        self._dense = []
        for prms in dense_params:
            self._dense.append(torch.nn.Linear(**prms))

    def set_acts(self, *, methods=None, parameters=None):

        self._acts = []

    def set_pool(self, *, parameters=None):
        # check if parameters are formatted correctly
        if not (isinstance(parameters, list) or len(parameters) < 1):
            raise ValueError(f"Pooling parameters were not formatted correctly: {parameters}")

        # instantiate parameter checker
        pool_checker = ParamChecker(name='pool', ikwiad=self._ikwiad)
        pool_checker.set_types(
            default={
                'kernel_size': 3,
                'stride': None,
                'padding': 0,
                'dilation': 1,
                'return_indices': False,
                'ceil_mode': False
            },
            dtypes={
                'kernel_size': (int, tuple),
                'stride': (types.NoneType, int, tuple),
                'padding': (int, tuple),
                'dilation': (int, tuple),
                'return_indices': (bool, int),
                'ceil_mode': (bool, int)
            },
            vtypes={
                'kernel_size': lambda x: (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'stride': lambda x: (isinstance(x, types.NoneType)) or (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'padding': lambda x: (isinstance(x, int) and 0 <= x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 <= i for i in x)),
                'dilation': lambda x: (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'return_indices': lambda x: True,
                'ceil_mode': lambda x: True
            },
            ctypes={
                'kernel_size': lambda x: x,
                'stride': lambda x: x,
                'padding': lambda x: x,
                'dilation': lambda x: x,
                'return_indices': lambda x: bool(x),
                'ceil_mode': lambda x: bool(x)
            }
        )

        # get each layer's parameters
        pool_params = []
        for prm in parameters:
            pool_params.append(pool_checker.check_params(prm))

        # set pooling layers
        self._pool = []
        for prms in pool_params:
            self._pool.append(torch.nn.MaxPool2d(**prms))

    def configure_network(self, loader):
        ...

    def set_hyperparameters(self):
        ...

    def forward(self, x):
        for cnv in range(len(self._conv)):
            x = self.acts[cnv](self._dense[cnv](x))
        for dns in range(len(self._conv), len(self._dense)):
            x = self.acts[dns](self._dense[dns](x))
        return x

    def fit(self, loader, *, parameters, **kwargs):
        hyperparam_checker = ParamChecker(name='hyperparams', ikwiad=self._ikwiad)
        hyperparam_checker.set_types(
            default={
                'epochs': 5
            },
            dtypes={
                'epochs': int
            },
            vtypes={
                'epochs': lambda x: 0 < x
            },
            ctypes={
                'epochs': lambda x: x
            }
        )
        params = hyperparam_checker.check_params(parameters, **kwargs)
        for epoch in range(params['epochs']):
            running_loss = 0.0
            for batch, (image, labels) in enumerate(loader, 0):
                self._optim.zero_grad()
                outputs = self.forward(batch)
                loss = self._loss(outputs, labels)
                loss.backward()
                self._optim.step()
                running_loss += loss.item()
                # todo: status bar
