r"""
**Core CNN.**

Attributes:
----------
**CNNCore**:
    Core CNN architecture.
"""

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
        self._instantiated = False
        self._instantiations = {
            ...  # todo: add this checker
        }
        # network features
        self._in_hgt = None
        self._in_wth = None
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

    def set_int_sizes(self, *, conv_channels: list = None, dense_sizes: list = None):
        # reset size lists
        # todo: check ordering
        self._conv_sizes = []
        self._dense_sizes = []
        if conv_channels is None:
            # default convolutional channels
            conv_channels = [16, 32, 64, 64]
        if dense_sizes is None:
            # default dense sizes
            dense_sizes = [128, 64, 32]
        # unpack sizes
        if not self._conv_sizes:
            # reset conv sizes
            self._conv_sizes = ['colors', *conv_channels]
        else:
            # add conv sizes
            self._conv_sizes += conv_channels
        if not self._dense_sizes:
            # reset dense sizes
            self._dense_sizes = ['flattened', *dense_sizes, 'classes']
        else:
            # add dense sizes
            self._dense_sizes = ['flattened'] + dense_sizes + self._dense_sizes

    def transfer_training_params(self, color_channels: int = None, classes: int = None, initial_height: int = None, initial_width: int = None, *, loader: DataLoader = None):
        if isinstance(loader, DataLoader):
            # transfer from loader
            # todo: add direct transferring of training parameters from dataloader
            raise RuntimeError("Direct transfer of training parameters from a dataloader is currently not implemented")
        else:
            # manual set
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
                self._in_hgt = initial_height
            else:
                # invalid initial height
                raise TypeError(
                    f"'initial_height' is invalid: {initial_height}",
                    f"'initial_height' must be a positive integer"
                )

            if isinstance(initial_width, int) and 0 < initial_width:
                # set internal initial width
                self._in_wth = initial_width
            else:
                # invalid initial width
                raise TypeError(
                    f"'initial_width' is invalid: {initial_width}",
                    f"'initial_width' must be a positive integer"
                )

    def set_acts(self, *, methods: list = None, parameters: list = None):
        # activation parameter reference
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
                    'dim': (type(None), int)
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

        # torch activation reference
        activation_ref = {
            'ReLU': torch.nn.ReLU,
            'Softplus': torch.nn.Softplus,
            'Softmax': torch.nn.Softmax,
            'Tanh': torch.nn.Tanh,
            'Sigmoid': torch.nn.Sigmoid,
            'Mish': torch.nn.Mish
        }

        if methods is not None:
            # set activations
            if len(methods) != len(parameters):
                # invalid match
                raise RuntimeError(
                    f"Invalid matching of 'params' and 'methods' ({len(methods)} != {len(parameters)})"
                    f"methods: {methods}"
                    f"parameters: {parameters}"
                )

            if not ((isinstance(methods, list) and isinstance(parameters, list)) and (len(parameters) != len(self._dense_sizes) + len(self._conv_sizes) - 1)):
                # invalid parameter format
                if not (isinstance(methods, list) and isinstance(parameters, list)):
                    # invalid type
                    raise ValueError(
                        f"'methods' and/or 'parameters' were not formatted correctly: {parameters}"
                        f"'methods' and/or 'parameters' must be a list"
                    )
                if not len(parameters) != len(self._dense_sizes) + len(self._conv_sizes) - 1:
                    # invalid length
                    raise ValueError(
                        f"'methods' and/or 'parameters' were not formatted correctly: {parameters}"
                        f"'methods' and/or 'parameters' must correspond with the amount of layers in the network ({len(self._dense_sizes) + len(self._conv_sizes) - 1})"
                    )

            if not all([mth in self.allowed_acts for mth in methods]):
                # invalid activator
                detected_invalid = []
                for mth in methods:
                    if mth not in self.allowed_acts:
                        detected_invalid.append(mth)
                raise ValueError(
                    f"Invalid methods detected: {detected_invalid}"
                    f"Choose from: {self.allowed_acts}"
                )
        else:
            # set default activations
            methods = ['ReLU'] * (len(self._dense_sizes) + len(self._conv_sizes) - 2)
            methods.append('Softmax')
            parameters = [{}] * (len(self._dense_sizes) + len(self._conv_sizes) - 1)

        for act_pair in zip(methods, parameters):
            # set activation objects
            act_checker = ParamChecker(name='activations', ikwiad=self._ikwiad)
            act_checker.set_types(**act_params[act_pair[0]])
            act_prms = act_checker.check_params(act_pair[1])
            self._acts.append(activation_ref[act_pair[0]](act_prms))

    def set_conv(self, *, parameters: list = None):
        if self._conv_sizes is None or not isinstance(self._conv_sizes[0], int):
            # invalid conv sizes
            raise RuntimeError(f"conv channels weren't set properly: {self._conv_sizes}")
        if parameters is not None:
            if not (isinstance(parameters, list)) or (len(parameters) != len(self._conv_sizes)):
                # invalid parameter format
                if not isinstance(parameters, list):
                    # invalid type
                    raise TypeError(
                        f"'parameters' for conv were not formatted correctly: {parameters}"
                        f"'parameters' must be a list"
                    )
                if len(parameters) != len(self._conv_sizes):
                    # invalid length
                    raise ValueError(
                        f"'parameters' for conv were not formatted correctly: {parameters}"
                        f"'parameters' don't match sizes ({len(parameters)} != {len(self._conv_sizes)})"
                    )
        else:
            # set default conv parameters
            parameters = [{}] * len(self._conv_sizes)

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

        # reset conv params
        self._conv_params = []
        for prm in parameters:
            # check conv params
            self._conv_params.append(conv_checker.check_params(prm))

    def set_pool(self, *, parameters: list = None):
        if self._conv_sizes is None or not isinstance(self._conv_sizes[0], int):
            # invalid conv sizes
            raise RuntimeError(f"conv channels weren't set properly: {self._conv_sizes}")
        if parameters is not None:
            if not (isinstance(parameters, list)) or (len(parameters) != len(self._conv_sizes)):
                # invalid parameter format
                if not isinstance(parameters, list):
                    # invalid type
                    raise TypeError(
                        f"'parameters' for pool were not formatted correctly: {parameters}"
                        f"'parameters' must be a list"
                    )
                if len(parameters) != len(self._conv_sizes):
                    # invalid length
                    raise ValueError(
                        f"'parameters' for pool were not formatted correctly: {parameters}"
                        f"'parameters' don't match sizes ({len(parameters)} != {len(self._conv_sizes)})"
                    )
        else:
            # set default pool parameters
            parameters = [{}] * len(self._conv_sizes)

        # instantiate parameter checker
        pool_checker = ParamChecker(name='Pooling Parameters', ikwiad=self._ikwiad)
        pool_checker.set_types(
            default={
                'kernel_size': 3,
                'stride': 1,
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

        # reset pool params
        self._pool_params = []
        for prm in parameters:
            # check pool params
            self._pool_params.append(pool_checker.check_params(prm))

    def set_dense(self, *, parameters: list = None):
        if self._dense_sizes is None:
            # invalid dense sizes
            raise RuntimeError(f"dense channels weren't set: {self._dense_sizes}")
        if parameters is not None:
            if not (isinstance(parameters, list)) or (len(parameters) != len(self._dense_sizes)):
                # invalid parameter format
                if not isinstance(parameters, list):
                    # invalid type
                    raise TypeError(
                        f"'parameters' for dense were not formatted correctly: {parameters}"
                        f"'parameters' must be a list"
                    )
                if len(parameters) != len(self._dense_sizes):
                    # invalid length
                    raise ValueError(
                        f"'parameters' for dense were not formatted correctly: {parameters}"
                        f"'parameters' don't match sizes ({len(parameters)} != {len(self._dense_sizes)})"
                    )
        else:
            # set default dense parameters
            parameters = [{}] * len(self._dense_sizes)

        # instantiate parameter checker
        dense_checker = ParamChecker(name='Dense Parameters', ikwiad=self._ikwiad)
        dense_checker.set_types(
            default={
                'bias': True
            },
            dtypes={
                'bias': (bool, int)
            },
            vtypes={
                'bias': lambda x: True
            },
            ctypes={
                'bias': lambda x: bool(x)
            }
        )

        # reset dense params
        self._dense_params = []
        for prm in parameters:
            # check dense params
            self._dense_params.append(dense_checker.check_params(prm))

    @staticmethod
    def _calc_lyr_size(h_in: int, w_in: int, params: dict):
        for key in params:
            # reformat parameters
            if not isinstance(params[key], list):
                params[key] = (params[key], params[key])
            else:
                params[key] = params[key]

        # out size calculation
        h_out = (h_in + 2 * params['padding'][0] - params['dilation'][0] * (params['kernel_size'][0] - 1) - 1) // params['stride'][0] + 1
        w_out = (w_in + 2 * params['padding'][1] - params['dilation'][1] * (params['kernel_size'][1] - 1) - 1) // params['stride'][1] + 1
        # return out size
        return h_out, w_out

    def instantiate_model(self):
        if self._conv_params is None:
            # invalid conv params
            raise RuntimeError(f"'conv parameters' weren't set: {self._conv_params}")
        if self._pool_params is None:
            # invalid pool params
            raise RuntimeError(f"'pool parameters' weren't set: {self._pool_params}")
        if self._dense_params is None:
            # invalid dense params
            raise RuntimeError(f"'dense parameters' weren't set: {self._dense_params}")
        if self._acts is None:
            raise RuntimeError(f"'activation parameters' weren't set: {self._acts}")

        # calculate flattened size
        final_hgt = self._in_hgt
        final_wth = self._in_wth
        for lyr in range(len(self._conv_params)):
            # grab necessary parameters
            calc_conv_params = {
                'padding': self._conv_params[lyr]['padding'],
                'dilation': self._conv_params[lyr]['dilation'],
                'kernel_size': self._conv_params[lyr]['kernel_size'],
                'stride': self._conv_params[lyr]['stride']
            }
            calc_pool_params = {
                'padding': self._pool_params[lyr]['padding'],
                'dilation': self._pool_params[lyr]['dilation'],
                'kernel_size': self._pool_params[lyr]['kernel_size'],
                'stride': self._pool_params[lyr]['stride']
            }
            if calc_pool_params['stride'] is None:
                # adjust stride for pool
                calc_pool_params['stride'] = calc_pool_params['kernel_size']
            final_hgt, final_wth = self._calc_lyr_size(final_hgt, final_wth, calc_conv_params)
            final_hgt, final_wth = self._calc_lyr_size(final_hgt, final_wth, calc_pool_params)
            print(final_hgt, final_wth)

        if final_hgt * final_wth <= 0:
            # check for zero instantiation
            raise ValueError(f"flattened size cannot be 0: {final_hgt * final_wth}")
        # set flattened size
        self._dense_sizes[0] = int(final_hgt * final_wth)
        # error reset rq
        self._dense_sizes[0] = 63488

        for prms in range(len(self._conv_params) - 1):
            # set conv layers
            self._conv.append(torch.nn.Conv2d(*self._conv_sizes[prms:prms + 2], **self._conv_params[prms]))
        for prms in self._pool_params:
            # set pooling
            self._pool.append(torch.nn.MaxPool2d(**prms))
        for prms in range(len(self._dense_params) - 1):
            # set dense layers
            self._dense.append(torch.nn.Linear(*self._dense_sizes[prms:prms + 2], **self._dense_params[prms]))

        # signal instantiation
        self._instantiated = True

    def check_instantiation(self):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(self._acts)
        print(self._conv)
        print(self._pool)
        print(self._dense)
        print(x.shape)
        if not self._instantiated:
            # invalid instantiation (this a really stupid way to implement this. too bad!)
            raise RuntimeError("Network was not fully instantiated")
        for cnv in range(len(self._conv)):
            print(self._acts[cnv])
            print(self._conv[cnv])
            # run through conv
            x = self._acts[cnv](self._conv[cnv](x))
        # flatten
        x.flatten()
        print(x.shape)
        for dns in range(len(self._dense)):
            # run through dense
            print(self._acts[dns + len(self._conv)])
            print(self._dense[dns])
            x = self._acts[dns + len(self._conv)](self._dense[dns](x))
        # return output
        return x
