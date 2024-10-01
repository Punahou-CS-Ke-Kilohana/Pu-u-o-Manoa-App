import types

import torch
import torch.functional as f
import torch.optim as optim
import torch.nn as nn

from .utils import ParamChecker


class TorchCNN(torch.nn.Module):
    def __init__(self, *, status_bars: bool = False, ikwiad: bool = False):
        super(TorchCNN, self).__init__()

        # internal visual settings
        self._ikwiad = ikwiad
        self._status = status_bars

        # network allowed settings
        self.allowed_acts = [
            'ReLU',
            'Softplus',
            'Softmax',
            'Tanh',
            'Sigmoid',
            'Mish'
        ]
        self.allowed_losses = [
            'CrossEntropyLoss'
            'MSELoss'
        ]
        self.allowed_optims = [
            'Adam',
            'AdamW',
            'Adagrad',
            'RMSprop',
            'SGD'
        ]

        # network features
        self._conv_sizes = None
        self._dense_sizes = None
        self._feature_space = None
        self._classifications = None

        # technical network elements
        self._acts = None
        self._conv = None
        self._pool = None
        self._dense = None
        self._optim = None
        self._loss = None

    def set_sizes(self, *, conv_channels: list = None, dense_sizes: list = None) -> None:
        if conv_channels is None:
            conv_channels = [128, 64, 32]
        if dense_sizes is None:
            dense_sizes = [256, 128, 64, 32]
        self._conv_sizes = ['batching', *conv_channels]
        self._dense_sizes = ['flattened', *dense_sizes, 'classes']

    def set_conv(self, *, parameters: list) -> None:
        if not (isinstance(parameters, list) or len(parameters) != len(self._conv_sizes) - 1):
            # invalid parameter format
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

    def set_pool(self, *, parameters: dict = None):
        if not (isinstance(parameters, dict) or len(parameters) < len(self._conv_sizes)):
            # invalid parameter format
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
                'kernel_size': lambda x: (isinstance(x, int) and 0 < x) or (
                            isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'stride': lambda x: (isinstance(x, types.NoneType)) or (isinstance(x, int) and 0 < x) or (
                            isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'padding': lambda x: (isinstance(x, int) and 0 <= x) or (
                            isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 <= i for i in x)),
                'dilation': lambda x: (isinstance(x, int) and 0 < x) or (
                            isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
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

    def set_dense(self, *, parameters=None):
        if not (isinstance(parameters, list) or len(parameters) != len(self._dense_sizes) - 1):
            # invalid parameter format
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
        # todo
        self._acts = []
        if len(methods) != len(parameters):
            raise RuntimeError("Not matching params and methods")
        if not all([mth in self.allowed_acts for mth in methods]):
            raise ValueError("Not a valid activator")

        act_params = {
            'ReLU': {
                'default': {
                    'inplace': False
                },
                'dtypes': {
                    'inplace': (bool, int)
                },
                'vtypes': {
                    'inplace': lambda x: True
                },
                'ctypes': {
                    'inplace': lambda x: bool(x)
                }
            },
            'Softplus': {
                'default': {
                    'beta': 1.0,
                    'threshold': 20.0
                },
                'dtypes': {
                    'beta': (float, int),
                    'threshold': (float, int)
                },
                'vtypes': {
                    'beta': lambda x: 0 < x,
                    'threshold': lambda x: 0 < x
                },
                'ctypes': {
                    'beta': lambda x: float(x),
                    'threshold': lambda x: float(x)
                }
            },
            'Softmax': {
                'default': {
                    'dim': None
                },
                'dtypes': {
                    'dim': (types.NoneType, int)
                },
                'vtypes': {
                    'dim': lambda x: True
                },
                'ctypes': {
                    'dim': lambda x: x
                }
            },
            'Tanh': {
                'default': None,
                'dtypes': None,
                'vtypes': None,
                'ctypes': None
            },
            'Sigmoid': {
                'default': None,
                'dtypes': None,
                'vtypes': None,
                'ctypes': None
            },
            'Mish': {
                'default': {
                    'inplace': False
                },
                'dtypes': {
                    'inplace': (bool, int)
                },
                'vtypes': {
                    'inplace': lambda x: True
                },
                'ctypes': {
                    'inplace': lambda x: bool(x)
                }
            }
        }

        # make activations reference
        activation_ref = {
            'ReLU': torch.nn.ReLU,
            'Softplus': torch.nn.Softplus,
            'Softmax': torch.nn.Softmax,
            'Tanh': torch.nn.Tanh,
            'Sigmoid': torch.nn.Sigmoid,
            'Mish': torch.nn.Mish
        }

        for act_pair in zip(methods, parameters):
            # set activation objects
            act_checker = ParamChecker(name='activations', ikwiad=self._ikwiad)
            act_checker.set_types(**act_params[act_pair[0]])
            act_prms = act_checker.check_params(act_pair[1])
            self._acts.append(activation_ref[[act_pair[0]](act_prms))

    def set_loss(self, method: str = 'CrossEntropyLoss', *, parameters: dict = None, **kwargs):
        def_loss_params = {
            'CrossEntropyLoss': {
                'default': {
                    'weight': None,
                    'size_average': True,
                    'ignore_index': -100,
                    'reduce': True,
                    'reduction': 'mean',
                    'label_smoothing': 0.0
                },
                'dtypes': {
                    'weight': (type(None), torch.Tensor),
                    'size_average': bool,
                    'ignore_index': int,
                    'reduce': bool,
                    'reduction': str,
                    'label_smoothing': float
                },
                'vtypes': {
                    'weight': lambda x: x is None or isinstance(x, torch.Tensor),
                    'size_average': lambda x: isinstance(x, bool),
                    'ignore_index': lambda x: isinstance(x, int),
                    'reduce': lambda x: isinstance(x, bool),
                    'reduction': lambda x: x in ['none', 'mean', 'sum'],
                    'label_smoothing': lambda x: 0.0 <= x <= 1.0
                },
                'ctypes': {
                    'weight': lambda x: x if x is None else torch.tensor(x, dtype=torch.float32),
                    'size_average': lambda x: bool(x),
                    'ignore_index': lambda x: int(x),
                    'reduce': lambda x: bool(x),
                    'reduction': lambda x: str(x),
                    'label_smoothing': lambda x: float(x)
                }
            },
            'MSELoss': {
                'default': {
                    'size_average': True,
                    'reduce': True,
                    'reduction': 'mean'
                },
                'dtypes': {
                    'size_average': bool,
                    'reduce': bool,
                    'reduction': str
                },
                'vtypes': {
                    'size_average': lambda x: isinstance(x, bool),
                    'reduce': lambda x: isinstance(x, bool),
                    'reduction': lambda x: x in ['none', 'mean', 'sum']
                },
                'ctypes': {
                    'size_average': lambda x: bool(x),
                    'reduce': lambda x: bool(x),
                    'reduction': lambda x: str(x)
                }
            },
            'L1Loss': {
                'default': {
                    'size_average': True,
                    'reduce': True,
                    'reduction': 'mean'
                },
                'dtypes': {
                    'size_average': bool,
                    'reduce': bool,
                    'reduction': str
                },
                'vtypes': {
                    'size_average': lambda x: isinstance(x, bool),
                    'reduce': lambda x: isinstance(x, bool),
                    'reduction': lambda x: x in ['none', 'mean', 'sum']
                },
                'ctypes': {
                    'size_average': lambda x: bool(x),
                    'reduce': lambda x: bool(x),
                    'reduction': lambda x: str(x)
                }
            }
        }

        # define pytorch loss reference
        loss_ref = {
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
            'MSELoss': torch.nn.MSELoss,
            'L1Loss': torch.nn.L1Loss
        }

        if method not in self.allowed_losses:
            # invalid loss method
            raise ValueError(
                f"Optimization method is invalid: {method}\n",
                f"Choose from: {[loss_mthd for loss_mthd in self.allowed_losses]}"
            )

        # set loss
        loss_checker = ParamChecker(name='losses', ikwiad=self._ikwiad)
        loss_checker.set_types(**def_loss_params[method])
        loss_params = loss_checker.check_params(parameters, **kwargs)
        self._loss = loss_ref[method](loss_params)

    def set_optim(self, method: str = 'Adam', *, parameters: dict = None, **kwargs):
        # define default optimization parameters
        def_optim_params = {
            'Adam': {
                'default': {
                    'lr': 0.001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0.0,
                    'amsgrad': False,
                    'foreach': None,
                    'maximize': False,
                    'fused': False
                },
                'dtypes': {
                    'lr': (float, int),
                    'betas': tuple,
                    'eps': (float, int),
                    'weight_decay': (float, int),
                    'amsgrad': (bool, int),
                    'foreach': (bool, int),
                    'maximize': (bool, int),
                    'fused': (bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0.0 <= x,
                    'betas': lambda x: len(x) == 2 and all(isinstance(i, float) and 0.0 < i < 1.0 for i in x),
                    'eps': lambda x: 0.0 < x,
                    'weight_decay': lambda x: 0.0 <= x,
                    'amsgrad': lambda x: True,
                    'foreach': lambda x: True,
                    'maximize': lambda x: True,
                    'fused': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'betas': lambda x: x,
                    'eps': lambda x: float(x),
                    'weight_decay': lambda x: float(x),
                    'amsgrad': lambda x: bool(x),
                    'foreach': lambda x: bool(x),
                    'maximize': lambda x: bool(x),
                    'fused': lambda x: bool(x)
                }
            },
            'AdamW': {
                'default': {
                    'lr': 0.001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0.0,
                    'amsgrad': False,
                    'foreach': None,
                    'maximize': False,
                    'fused': False
                },
                'dtypes': {
                    'lr': (float, int),
                    'betas': tuple,
                    'eps': (float, int),
                    'weight_decay': (float, int),
                    'amsgrad': (bool, int),
                    'foreach': (bool, int),
                    'maximize': (bool, int),
                    'fused': (bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0.0 <= x,
                    'betas': lambda x: len(x) == 2 and all(isinstance(i, float) and 0.0 < i < 1.0 for i in x),
                    'eps': lambda x: 0.0 < x,
                    'weight_decay': lambda x: 0.0 <= x,
                    'amsgrad': lambda x: True,
                    'foreach': lambda x: True,
                    'maximize': lambda x: True,
                    'fused': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'betas': lambda x: x,
                    'eps': lambda x: float(x),
                    'weight_decay': lambda x: float(x),
                    'amsgrad': lambda x: bool(x),
                    'foreach': lambda x: bool(x),
                    'maximize': lambda x: bool(x),
                    'fused': lambda x: bool(x)
                }
            },
            'Adagrad': {
                'default': {
                    'lr:': 0.01,
                    'lr_decay': 0.0,
                    'weight_decay': 0.0,
                    'initial_accumulator_value': 0.0,
                    'eps': 1e-10,
                    'foreach': None,
                    'maximize': False,
                    'fused': None
                },
                'dtypes': {
                    'lr': (float, int),
                    'lr_decay': (float, int),
                    'weight_decay': (float, int),
                    'initial_accumulator_value': (float, int),
                    'eps': (float, int),
                    'foreach': (bool, int),
                    'maximize': (bool, int),
                    'fused': (bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0.0 <= x,
                    'lr_decay': lambda x: 0.0 <= x,
                    'weight_decay': lambda x: 0.0 <= x,
                    'initial_accumulator_value': lambda x: 0.0 <= x,
                    'eps': lambda x: 0.0 < x,
                    'foreach': lambda x: True,
                    'maximize': lambda x: True,
                    'fused': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'lr_decay': lambda x: float(x),
                    'weight_decay': lambda x: float(x),
                    'initial_accumulator_value': lambda x: float(x),
                    'eps': lambda x: float(x),
                    'foreach': lambda x: bool(x),
                    'maximize': lambda x: bool(x),
                    'fused': lambda x: bool(x)
                }
            },
            'RMSprop': {
                'default': {
                    'lr:': 0.01,
                    'alpha': 0.99,
                    'eps': 1e-10,
                    'weight_decay': 0.0,
                    'momentum': 0.0,
                    'centered': False,
                    'foreach': None,
                    'maximize': False
                },
                'dtypes': {
                    'lr': (float, int),
                    'alpha': (float, int),
                    'eps': (float, int),
                    'weight_decay': (float, int),
                    'momentum': (float, int),
                    'centered': (bool, int),
                    'foreach': (bool, int),
                    'maximize': (bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0.0 <= x,
                    'alpha': lambda x: 0.0 < x < 1.0,
                    'eps': lambda x: 0.0 < x,
                    'weight_decay': lambda x: 0.0 <= x,
                    'momentum': lambda x: 0.0 <= x < 1.0,
                    'centered': lambda x: True,
                    'foreach': lambda x: True,
                    'maximize': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'alpha': lambda x: float(x),
                    'eps': lambda x: float(x),
                    'weight_decay': lambda x: float(x),
                    'momentum': lambda x: float(x),
                    'centered': lambda x: bool(x),
                    'foreach': lambda x: bool(x),
                    'maximize': lambda x: bool(x)
                }
            },
            'SGD': {
                'default': {
                    'lr:': 0.001,
                    'momentum': 0.0,
                    'dampening': 0.0,
                    'weight_decay': 0.0,
                    'nesterov': False,
                    'maximize': False,
                    'foreach': None,
                    'fused': None
                },
                'dtypes': {
                    'lr:': (float, int),
                    'momentum': (float, int),
                    'dampening': (float, int),
                    'weight_decay': (float, int),
                    'nesterov': (bool, int),
                    'maximize': (bool, int),
                    'foreach': (bool, int),
                    'fused': (bool, int)
                },
                'vtypes': {
                    'lr:': lambda x: 0.0 <= x,
                    'momentum': lambda x: 0.0 <= x < 1.0,
                    'dampening': lambda x: 0.0 <= x < 1.0,
                    'weight_decay': lambda x: 0.0 <= x,
                    'nesterov': lambda x: x,
                    'maximize': lambda x: x,
                    'foreach': lambda x: x,
                    'fused': lambda x: x
                },
                'ctypes': {
                    'lr:': lambda x: float(x),
                    'momentum': lambda x: float(x),
                    'dampening': lambda x: float(x),
                    'weight_decay': lambda x: float(x),
                    'nesterov': lambda x: bool(x),
                    'maximize': lambda x: bool(x),
                    'foreach': lambda x: bool(x),
                    'fused': lambda x: bool(x)
                }
            }
        }

        # define pytorch optimization reference
        optim_ref = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'Adagrad': torch.optim.Adagrad,
            'RMSprop': torch.optim.RMSprop,
            'SGD': torch.optim.SGD
        }

        if method not in self.allowed_optims:
            # invalid optimization method
            raise ValueError(
                f"Optimization method is invalid: {method}\n",
                f"Choose from: {[optim_mthd for optim_mthd in self.allowed_optims]}"
            )

        # set optimizer
        optim_checker = ParamChecker(name='optimizer', ikwiad=self._ikwiad)
        optim_checker.set_types(**def_optim_params[method])
        optim_params = optim_checker.check_params(parameters, **kwargs)
        self._optim = optim_ref[method](optim_params)

    def configure_network(self, loader):
        # todo
        ...

    def set_hyperparameters(self):
        # todo
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
