import types

import torch

from application.utils import ParamChecker
from application.dataloader import DataLoader  # not sure what happened with this, but it's fine for now


class TorchCNNCore(torch.nn.Module):
    def __init__(self, *, ikwiad: bool = False):
        super(TorchCNNCore, self).__init__()

        # allowed methods
        # activations
        self.allowed_acts = [
            'ReLU',
            'Softplus',
            'Softmax',
            'Tanh',
            'Sigmoid',
            'Mish'
        ]

        # internals
        # error messages
        self._ikwiad = ikwiad
        # proper setup checker
        self._instantiated = False
        # initial dimensions
        self._in_hgt = None
        self._in_wth = None
        # activations
        self._acts = []
        # network features
        self._conv_sizes = []
        self._dense_sizes = []
        # convolution
        self._conv = []
        self._conv_calcs = []
        # pool
        self._pool = []
        self._pool_calcs = []
        # dense
        self._dense = []

    def transfer_training_params(self, batching: int, classes: int, initial_height: int, initial_width: int, *, loader: DataLoader = None):
        if loader is not None:
            # transfer from loader
            raise RuntimeError("Direct transfer of training parameters from a dataloader is currently not implemented")
        else:
            # manual set
            if isinstance(batching, int) and 0 < batching:
                # set internal batching
                self._conv_sizes[0] = batching
            else:
                # invalid batching
                raise TypeError(
                    f"'batching' is invalid: {batching}",
                    f"'batching' must be a positive integer"
                )

            if isinstance(classes, int) and 0 < classes:
                # set internal classes
                self._dense_sizes[0] = classes
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

    def set_int_sizes(self, *, conv_channels: list = None, dense_sizes: list = None):
        if conv_channels is None:
            # default convolutional channels
            conv_channels = [128, 64, 32]
        if dense_sizes is None:
            # default dense sizes
            dense_sizes = [256, 128, 64, 32]
        # unpack sizes
        if not self._conv_sizes:
            # reset conv sizes
            self._conv_sizes = ['batching', *conv_channels]
        else:
            # add conv sizes
            self._conv_sizes += conv_channels
        if not self._dense_sizes:
            # reset dense sizes
            self._dense_sizes = ['flattened', *dense_sizes, 'classes']
        else:
            # add dense sizes
            self._dense_sizes = ['flattened'] + dense_sizes + self._dense_sizes

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
            methods = ['relu'] * (len(self._dense_sizes) + len(self._conv_sizes) - 2)
            methods.append('softmax')
            parameters = [] * (len(self._dense_sizes) + len(self._conv_sizes) - 1)

        for act_pair in zip(methods, parameters):
            # set activation objects
            act_checker = ParamChecker(name='activations', ikwiad=self._ikwiad)
            act_checker.set_types(**act_params[act_pair[0]])
            act_prms = act_checker.check_params(act_pair[1])
            self._acts.append(activation_ref[act_pair[0]](act_prms))

    def set_conv(self, *, parameters: list):
        if not (isinstance(parameters, list)) or (len(parameters) != len(self._conv_sizes)):
            # invalid parameter format
            if not isinstance(parameters, list):
                # invalid type
                raise TypeError(
                    f"'parameters' were not formatted correctly: {parameters}"
                    f"'parameters' must be a list"
                )
            if len(parameters) != len(self._conv_sizes):
                # invalid length
                raise ValueError(
                    f"'parameters' were not formatted correctly: {parameters}"
                    f"'parameters' don't match sizes ({len(parameters)} != {len(self._conv_sizes)})"
                )

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

        # check conv params
        conv_params = []
        for prm in parameters:
            conv_params.append(conv_checker.check_params(prm))

        # set convolutional layers
        for prms in range(len(conv_params)):
            # todo: this doesn't work  # why?
            self._conv.append(torch.nn.Conv2d(*self._conv_sizes[prms:prms + 1], **conv_params[prms]))

        # save calc conv params
        self._conv_calcs = []

    def set_pool(self, *, parameters: list):
        if not (isinstance(parameters, list)) or (len(parameters) != len(self._conv_sizes)):
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
        for prms in pool_params:
            self._pool.append(torch.nn.MaxPool2d(**prms))

    def set_dense(self, *, parameters: list):
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
        for prms in range(len(dense_params)):
            self._dense.append(torch.nn.Linear(*self._dense_sizes[prms:prms + 1], **dense_params[prms]))

    @staticmethod
    def _calc_lyr_size(h_in: int, w_in: int, params: dict):
        # params = {
        #     'padding': padding,
        #     'dilation': dilation,
        #     'kernel_size': kernel_size,
        #     'stride': stride
        # }
        for key in params:
            # reformat parameters
            if len(params[key]) == 1:
                params[key] = (params[key], params[key])
            else:
                params[key] = params[key]

        # out size calculation
        h_out = (h_in + 2 * params['padding'][0] - params['dilation'][0] * (params['kernel_size'][0] - 1) - 1) / params['stride'][0] + 1
        w_out = (w_in + 2 * params['padding'][1] - params['dilation'][1] * (params['kernel_size'][1] - 1) - 1) / params['stride'][1] + 1
        # return out size
        return h_out, w_out

    def forward(self, x):
        for cnv in range(len(self._conv)):
            # run through conv
            x = self.acts[cnv](self._dense[cnv](x))
        # flatten
        x.flatten()
        for dns in range(len(self._conv), len(self._dense)):
            # run through dense
            x = self.acts[dns](self._dense[dns](x))
        # return output
        return x
