from typing import Union

import torch

from ..utils.utils import ParamChecker
from ..utils.dataloader import DataLoader  # not sure what happened with this, but it's fine for now


class CNNCore(torch.nn.Module):
    def __init__(self, *, ikwiad: bool = False):
        super(CNNCore, self).__init__()

        # allowed activations list
        self._allowed_acts = [
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
            'dense': False,
            'fully_instantiated': False
        }
        # network features
        self._in_dims = None
        self._conv_sizes = None
        self._dense_sizes = None
        self._conv_params = None
        self._pool_params = None
        self._dense_params = None
        # activation container
        self._act_params = None
        self._conv_acts = torch.nn.ModuleList()
        self._dense_acts = torch.nn.ModuleList()
        # layer containers
        self._conv = torch.nn.ModuleList()
        self._pool = torch.nn.ModuleList()
        self._dense = torch.nn.ModuleList()

    @classmethod
    def get_allowed_acts(cls):
        return cls._allowed_acts

    @property
    def instantiations(self):
        return self._instantiations

    def set_channels(self, *, conv_channels: list = None, dense_channels: list = None) -> None:
        # check for duplicate initialization attempts
        assert not self._instantiations['channels'], "Channels can't be set twice"

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

    def transfer_training_params(self,
                                 color_channels: Union[int, None] = None,
                                 classes: Union[int, None] = None,
                                 initial_dims: Union[tuple, None] = None,
                                 *,
                                 loader: Union[DataLoader, None] = None) -> None:
        # check for channel set
        assert self._instantiations['channels'], "Channels weren't set"
        # check for duplicate initialization attempts
        assert not self._instantiations['training_params'], "Training parameters can't be set twice"

        if isinstance(loader, DataLoader):
            # todo: add direct transferring of training parameters from dataloader
            raise RuntimeError("Direct transfer of training parameters from a dataloader is currently not implemented")
        else:
            assert isinstance(color_channels, int) and 0 < color_channels, "'color_channels' must be a positive integer"
            self._conv_sizes[0] = color_channels
            assert isinstance(classes, int) and 0 < classes, "'classes' must be a positive integer"
            self._dense_sizes[-1] = classes
            assert (isinstance(initial_dims, tuple) and
                    all([isinstance(itm, int) and itm for itm in initial_dims])
                    and len(initial_dims) == 2), \
                'initial dims must be a tuple of two positive integers'
            self._in_dims = initial_dims

        self._instantiations['training_params'] = True
        return None

    def set_acts(self, *, methods: Union[list, None] = None, parameters: Union[list, None] = None) -> None:
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

        # check for channel set
        assert self._instantiations['channels'], "Channels weren't set"
        # check for duplicate initialization attempts
        assert not self._instantiations['activators'], "Activators can't be set twice"

        if methods is None:
            # set default activators
            methods = ['ReLU'] * (len(self._dense_sizes) + len(self._conv_sizes) - 2)
            methods.append('Softmax')
            parameters = [{}] * (len(self._dense_sizes) + len(self._conv_sizes) - 1)
        else:
            # check for errors
            assert len(methods) == len(parameters) == (len(self._conv_sizes) + len(self._dense_sizes)), \
                (f"Invalid matching of 'params', 'methods', and channels\n"
                 f"({len(methods)} != {len(parameters)} != {len(self._conv_sizes) + len(self._dense_sizes)})")
            assert all([mth in self._allowed_acts for mth in methods]), \
                (f"Invalid methods detected in {methods}\n"
                 f"Choose from: {self._allowed_acts}")
            assert isinstance(methods, list), f"'methods' must be a list"
            assert isinstance(parameters, list), f"'parameters' must be a list"
            assert len(parameters) == len(self._dense_sizes) + len(self._conv_sizes) - 1, \
                (f"'methods' and/or 'parameters' must correspond with the amount of layers in the network\n"
                 f"({len(self._dense_sizes) + len(self._conv_sizes) - 1})")

        self._act_params = []
        for mthd, prms in zip(methods, parameters):
            # set activation objects
            act_checker = ParamChecker(name=f'Activator Parameters ({mthd})', ikwiad=self._ikwiad)
            act_checker.set_types(**act_params[mthd])
            self._act_params.append({
                'mthd': mthd,
                'prms': act_checker.check_params(prms)
            })

        self._instantiations['activators'] = True
        return None

    def set_conv(self, *, parameters: Union[list, None] = None) -> None:
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

        # check for channel set
        assert self._instantiations['channels'], "Channels weren't set"
        # check for duplicate initialization attempts
        assert not self._instantiations['convolutional'], "Convolutional layers can't be set twice"

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._conv_sizes) - 1)
        else:
            # check parameter list
            assert isinstance(parameters, list), f"'parameters' must be a list ({type(parameters)} != list)"
            assert len(parameters) == len(self._conv_sizes), \
                ("'parameters' length must match conv layers\n"
                 f"({len(parameters)} != {len(self._conv_sizes)})")

        # validate conv params
        self._conv_params = [conv_checker.check_params(prm) for prm in parameters]
        self._instantiations['convolutional'] = True
        return None

    def set_pool(self, *, parameters: Union[list, None] = None) -> None:
        # instantiate parameter checker
        pool_checker = ParamChecker(name='Pooling Parameters', ikwiad=self._ikwiad)
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

        # check for channel set
        assert self._instantiations['channels'], "Channels weren't set"
        # check for duplicate initialization attempts
        assert not self._instantiations['pooling'], "Pooling layers can't be set twice"

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._conv_sizes) - 1)
        else:
            # check parameter list
            assert isinstance(parameters, list), f"'parameters' must be a list ({type(parameters)} != list)"
            assert len(parameters) == len(self._conv_sizes), \
                ("'parameters' length must match conv layers\n"
                 f"({len(parameters)} != {len(self._conv_sizes)})")

        # validate pool params
        self._pool_params = [pool_checker.check_params(prm) if prm is not False else None for prm in parameters]
        self._instantiations['pooling'] = True
        return None

    def set_dense(self, *, parameters: Union[list, None] = None) -> None:
        # initialize parameter checker
        dense_checker = ParamChecker(name='Dense Parameters', ikwiad=self._ikwiad)
        dense_checker.set_types(
            default={'bias': True},
            dtypes={'bias': (bool, int)},
            vtypes={'bias': lambda x: True},
            ctypes={'bias': lambda x: bool(x)}
        )

        # check for channel set
        assert self._instantiations['channels'], "Channels weren't set"
        # check for duplicate initialization attempts
        assert not self._instantiations['dense'], "Dense layers can't be set twice"

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._dense_sizes) - 1)
        else:
            # check parameter list
            assert isinstance(parameters, list), f"'parameters' must be a list ({type(parameters)} != list)"
            assert len(parameters) == len(self._dense_sizes), \
                ("'parameters' length must match dense layers\n"
                 f"({len(parameters)} != {len(self._dense_sizes)})")

        # validate dense params
        self._dense_params = [dense_checker.check_params(prm) for prm in parameters]
        self._instantiations['dense'] = True
        return None

    @staticmethod
    def _calc_lyr_size(dims: tuple, params: dict):
        # parameter reformatting
        params = {key: (val, val) if isinstance(val, int) else val for key, val in params.items()}
        # out size calculation
        h_in, w_in = dims
        h_out = (h_in + 2 * params['padding'][0] - params['dilation'][0] * (params['kernel_size'][0] - 1) - 1) // params['stride'][0] + 1
        w_out = (w_in + 2 * params['padding'][1] - params['dilation'][1] * (params['kernel_size'][1] - 1) - 1) // params['stride'][1] + 1
        # return out size
        return h_out, w_out

    def instantiate_model(self, *, crossentropy: bool = True) -> None:
        # check for proper instantiation
        nec_instantiations = self._instantiations.copy()
        nec_instantiations.pop('fully_instantiated')
        assert all(nec_instantiations.values()), \
            (f"Necessary settings weren't fully instantiated:\n"
             f"{nec_instantiations}")

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
        assert 0 < final_size, "Flattened size cannot be 0"
        self._dense_sizes[0] = final_size

        for i, (conv, pool) in enumerate(zip(self._conv_params, self._pool_params)):
            # set conv and pool layers
            self._conv.append(torch.nn.Conv2d(*self._conv_sizes[i:i + 2], **conv))
            if pool is not None:
                self._pool.append(torch.nn.MaxPool2d(**pool))
            else:
                self._pool.append(torch.nn.Identity())
        for i, prms in enumerate(self._dense_params):
            # set dense layers
            self._dense.append(torch.nn.Linear(*self._dense_sizes[i:i + 2], **prms))

        # set forward method
        self.forward = self._compile_forward(crossentropy)
        self._instantiations['fully_instantiated'] = True
        return None

    def _compile_forward(self, crossentropy):
        # torch activation reference
        activation_ref = {
            'ReLU': torch.nn.ReLU,
            'Softplus': torch.nn.Softplus,
            'Softmax': torch.nn.Softmax,
            'Tanh': torch.nn.Tanh,
            'Sigmoid': torch.nn.Sigmoid,
            'Mish': torch.nn.Mish
        }

        # set act lists
        for pair in self._act_params[:len(self._conv) + 1]:
            self._conv_acts.append(activation_ref[pair['mthd']](pair['prms']))
        for pair in self._act_params[len(self._conv):]:
            self._dense_acts.append(activation_ref[pair['mthd']](pair['prms']))

        if crossentropy:
            def _forward(x: torch.Tensor) -> torch.Tensor:
                # set forward
                for act, conv, pool in zip(self._conv_acts, self._conv, self._pool):
                    x = pool(act(conv(x)))
                x = torch.flatten(x, 1)
                for act, dense in zip(self._dense_acts[:-1], self._dense[:-1]):
                    x = act(dense(x))
                return self._dense[-1](x)
        else:
            def _forward(x: torch.Tensor) -> torch.Tensor:
                # set forward
                for act, conv, pool in zip(self._conv_acts, self._conv, self._pool):
                    x = pool(act(conv(x)))
                x = torch.flatten(x, 1)
                for act, dense in zip(self._dense_acts, self._dense):
                    x = act(dense(x))
                return x

        return _forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # model not fully instantiated
        raise RuntimeError(f"Model wasn't fully instantiated\n({self._instantiations})")
