r"""
**Core CNN.**

Attributes:
----------
**CNNCore**:
    Core CNN architecture.
"""

from typing import Union

import torch

from application.utils import ParamChecker
from application.dataloader import DataLoader  # not sure what happened with this, but it's fine for now


class CNNCore(torch.nn.Module):
    # todo: add only single-parse permissions
    def __init__(self, *, ikwiad: bool = False):
        super(CNNCore, self).__init__()

        # allowed activations list
        self.allowed_acts = [
            'ReLU',
            'Softplus',
            'Softmax',
            'Tanh',
            'Sigmoid',
            'Mish'
        ]

        # internals
        # internal checkers
        self._ikwiad = ikwiad
        self._instantiations = {
            'channels': False,
            'training_params': False,
            'activators': False,
            'convolutional': False,
            'pooling': False,
            'dense': False
        }
        # network features
        self._in_dims = None
        self._conv_sizes = None
        self._dense_sizes = None
        self._conv_params = None
        self._pool_params = None
        self._dense_params = None
        # activation container
        self._acts = torch.nn.ModuleList()
        # layer containers
        self._conv = torch.nn.ModuleList()
        self._pool = torch.nn.ModuleList()
        self._dense = torch.nn.ModuleList()

    def set_channels(self, *, conv_channels: list = None, dense_channels: list = None) -> None:
        # check channel types
        assert ((isinstance(conv_channels, list) and all(isinstance(itm, int) for itm in conv_channels))
                or conv_channels is None), \
            "'conv_channels' weren't set correctly (list of integers or None)"
        assert ((isinstance(dense_channels, list) and all(isinstance(itm, int) for itm in dense_channels))
                or dense_channels is None), \
            "'dense_channels' weren't set correctly (list of integers or None)"

        # set channels
        conv_channels = conv_channels or [16, 32, 64, 128]
        dense_channels = dense_channels or [256, 128, 64, 32]
        # set internal channels
        self._conv_sizes = ['colors', *conv_channels]
        self._dense_sizes = ['flattened', *dense_channels, 'classes']
        self._instantiations['channels'] = True
        return None

    def transfer_training_params(self, color_channels: int = None, classes: int = None, initial_height: int = None, initial_width: int = None, *, loader: DataLoader = None):
        # todo: this segment of code is an absolute mess, and should eventually be nearly completely rewritten
        if isinstance(loader, DataLoader):
            # transfer from loader
            # todo: add direct transferring of training parameters from dataloader
            raise RuntimeError("Direct transfer of training parameters from a dataloader is currently not implemented")
        else:
            # manual set (done badly but hopefully will be improved)
            if isinstance(color_channels, int) and 0 < color_channels:
                # set internal initial channel
                self._conv_sizes[0] = color_channels
            else:
                # invalid colors
                raise TypeError(
                    f"'color_channels' is invalid: {color_channels}",
                    f"'color_channels' must be a positive integer"
                )

            if isinstance(classes, int) and 0 < classes:
                # set internal classes
                self._dense_sizes[-1] = classes
            else:
                # invalid classes
                raise TypeError(
                    f"'classes' is invalid: {classes}",
                    f"'classes' must be a positive integer"
                )

            if isinstance(initial_height, int) and 0 < initial_height:
                # set internal initial height
                in_hgt = initial_height
            else:
                # invalid initial height
                raise TypeError(
                    f"'initial_height' is invalid: {initial_height}",
                    f"'initial_height' must be a positive integer"
                )

            if isinstance(initial_width, int) and 0 < initial_width:
                # set internal initial width
                in_wth = initial_width
            else:
                # invalid initial width
                raise TypeError(
                    f"'initial_width' is invalid: {initial_width}",
                    f"'initial_width' must be a positive integer"
                )

            self._in_dims = (in_hgt, in_wth)
        self._instantiations['training_params'] = True

    def set_acts(self, *, methods: list = None, parameters: list = None):
        # torch activation reference
        activation_ref = {
            'ReLU': torch.nn.ReLU,
            'Softplus': torch.nn.Softplus,
            'Softmax': torch.nn.Softmax,
            'Tanh': torch.nn.Tanh,
            'Sigmoid': torch.nn.Sigmoid,
            'Mish': torch.nn.Mish
        }

        # activation parameter reference
        act_params = {
            'ReLU': {
                'default': {'inplace': False},
                'dtypes': {'inplace': (bool, int)},
                'vtypes': {'inplace': lambda x: True},
                'ctypes': {'inplace': lambda x: bool(x)}
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
                'default': {'dim': None},
                'dtypes': {'dim': (type(None), int)},
                'vtypes': {'dim': lambda x: True},
                'ctypes': {'dim': lambda x: x}
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
                'default': {'inplace': False},
                'dtypes': {'inplace': (bool, int)},
                'vtypes': {'inplace': lambda x: True},
                'ctypes': {'inplace': lambda x: bool(x)}
            }
        }

        if methods is None:
            # set default activators
            methods = ['ReLU'] * (len(self._dense_sizes) + len(self._conv_sizes) - 2)
            methods.append('Softmax')
            parameters = [{}] * (len(self._dense_sizes) + len(self._conv_sizes) - 1)
        else:
            # check for errors
            assert len(methods) == len(parameters), \
                f"Invalid matching of 'params' and 'methods' ({len(methods)} != {len(parameters)})"
            assert all([mth in self.allowed_acts for mth in methods]), \
                (f"Invalid methods detected in {methods}\n"
                 f"Choose from: {self.allowed_acts}")
            assert isinstance(methods, list), f"'methods' must be a list"
            assert isinstance(parameters, list), f"'parameters' must be a list"
            assert len(parameters) == len(self._dense_sizes) + len(self._conv_sizes) - 1, \
                (f"'methods' and/or 'parameters' must correspond with the amount of layers in the network\n"
                 f"({len(self._dense_sizes) + len(self._conv_sizes) - 1})")

        for i, (mthd, prms) in enumerate(zip(methods, parameters)):
            # set activation objects
            act_checker = ParamChecker(name=f'Activator Parameters ({i})', ikwiad=self._ikwiad)
            act_checker.set_types(**act_params[mthd])
            act_prms = act_checker.check_params(prms)
            self._acts.append(activation_ref[mthd](act_prms))

        self._instantiations['activators'] = True
        return None

    def set_conv(self, *, parameters: list = None):
        # check for channel set
        assert self._instantiations['channels'], "channels weren't set"

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._conv_sizes) - 1)
        else:
            # check parameter list
            assert isinstance(parameters, list), f"'parameters' must be a list ({type(parameters)} != list)"
            assert len(parameters) == len(self._conv_sizes), \
                ("'parameters' length must match conv layers\n"
                 f"({len(parameters)} != {len(self._conv_sizes)})")

        # instantiate parameter checker
        conv_checker = ParamChecker(name='Convolutional Parameters', ikwiad=self._ikwiad)
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

        # validate conv params
        self._conv_params = [conv_checker.check_params(prm) for prm in parameters]
        self._instantiations['convolutional'] = True
        return None

    def set_pool(self, *, parameters: Union[list, None] = None) -> None:
        # check for channel set
        assert self._instantiations['channels'], "channels weren't set"

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._conv_sizes) - 1)
        else:
            # check parameter list
            assert isinstance(parameters, list), f"'parameters' must be a list ({type(parameters)} != list)"
            assert len(parameters) == len(self._conv_sizes), \
                ("'parameters' length must match conv layers\n"
                 f"({len(parameters)} != {len(self._conv_sizes)})")

        # instantiate parameter checker
        pool_checker = ParamChecker(name='Pooling Parameters', ikwiad=self._ikwiad)
        pool_checker.set_types(
            default={
                'kernel_size': 3,
                'stride': 2,
                'padding': 0,
                'dilation': 1,
                'return_indices': False,
                'ceil_mode': False
            },
            dtypes={
                'kernel_size': (int, tuple),
                'stride': (type(None), int, tuple),
                'padding': (int, tuple),
                'dilation': (int, tuple),
                'return_indices': (bool, int),
                'ceil_mode': (bool, int)
            },
            vtypes={
                'kernel_size': lambda x: (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
                'stride': lambda x: (isinstance(x, type(None))) or (isinstance(x, int) and 0 < x) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x)),
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

        # validate pool params
        self._pool_params = [pool_checker.check_params(prm) for prm in parameters]
        self._instantiations['pooling'] = True
        return None

    def set_dense(self, *, parameters: Union[list, None] = None) -> None:
        # check for channel set
        assert self._instantiations['channels'], "channels weren't set"

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._dense_sizes) - 1)
        else:
            # check parameter list
            assert isinstance(parameters, list), f"'parameters' must be a list ({type(parameters)} != list)"
            assert len(parameters) == len(self._dense_sizes), \
                ("'parameters' length must match dense layers\n"
                 f"({len(parameters)} != {len(self._dense_sizes)})")

        # initialize parameter checker
        dense_checker = ParamChecker(name='Dense Parameters', ikwiad=self._ikwiad)
        dense_checker.set_types(
            default={'bias': True},
            dtypes={'bias': (bool, int)},
            vtypes={'bias': lambda x: True},
            ctypes={'bias': lambda x: bool(x)}
        )

        # validate dense params
        self._dense_params = [dense_checker.check_params(prm) for prm in parameters]
        self._instantiations['dense'] = True
        return None

    @staticmethod
    def _calc_lyr_size(dims: tuple, params: dict):
        # parameter reformatting
        params = {key: (val, val) if not isinstance(val, list) else val for key, val in params.items()}
        # out size calculation
        h_in, w_in = dims
        h_out = (h_in + 2 * params['padding'][0] - params['dilation'][0] * (params['kernel_size'][0] - 1) - 1) // params['stride'][0] + 1
        w_out = (w_in + 2 * params['padding'][1] - params['dilation'][1] * (params['kernel_size'][1] - 1) - 1) // params['stride'][1] + 1
        # return out size
        return h_out, w_out

    def instantiate_model(self) -> None:
        # check for proper instantiation
        assert all(self._instantiations.values()), \
            (f"model wasn't fully instantiated:\n"
             f"{self._instantiations}")

        dims = self._in_dims
        for conv, pool in zip(self._conv_params, self._pool_params):
            # grab necessary parameters
            calc_conv_params = {
                'padding': conv['padding'],
                'dilation': conv['dilation'],
                'kernel_size': conv['kernel_size'],
                'stride': conv['stride']
            }
            calc_pool_params = {
                'padding': pool['padding'],
                'dilation': pool['dilation'],
                'kernel_size': pool['kernel_size'],
                'stride': pool['kernel_size'] if pool['stride'] is None else pool['stride']
            }
            dims = self._calc_lyr_size(dims, calc_conv_params)
            dims = self._calc_lyr_size(dims, calc_pool_params)

        # set flattened size
        h, w = dims
        final_size = int(h * w * self._conv_sizes[-1])
        assert 0 < final_size, "flattened size cannot be 0"
        self._dense_sizes[0] = final_size

        for i, (conv, pool) in enumerate(zip(self._conv_params, self._pool_params)):
            # set conv and pool layers
            self._conv.append(torch.nn.Conv2d(*self._conv_sizes[i:i + 2], **conv))
            self._pool.append(torch.nn.MaxPool2d(**pool))
        for i, prms in enumerate(self._dense_params):
            # set dense layers
            self._dense.append(torch.nn.Linear(*self._dense_sizes[i:i + 2], **prms))
        return None

    # def _compile_forward(self):
    #     # todo
    #     def _conv(acts, convs, pools, x):
    #         for act, conv, pool in zip(acts, convs, pools):
    #             x = pool(act(conv(x)))
    #         return x
    #
    #     def _dense(acts, denses, x):
    #         for act, dense in zip(acts, denses):
    #             x = act(dense(x))
    #         return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (conv, pool) in enumerate(zip(self._conv, self._pool)):
            # run through conv and pool
            x = pool(self._acts[i](conv(x)))
        # flatten
        x = torch.flatten(x, 1)
        for i, dense in enumerate(self._dense[:-1]):
            # run through dense
            x = self._acts[i + len(self._conv)](dense(x))
        x = self._dense[-1](x)
        # return output
        return x
