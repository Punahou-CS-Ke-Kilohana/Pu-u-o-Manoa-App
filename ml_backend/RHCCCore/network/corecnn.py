r"""
**Pu-u-Manoa-App Core CNN.**

Contains:
    - :class:`CNNCore`
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from warnings import warn

from ..utils.utils import Params, ParamChecker
from ..utils.errors import (
    AlreadySetError,
    MissingMethodError
)


class CNNCore(nn.Module):
    r"""
    **Main configurable CNN Object.**

    A configurable Convolutional Neural Network (CNN) based on PyTorch.
    CNNCore is the core component of a configurable Convolutional Neural Network (CNN) model.

    Note:
        CNNCore automatically assembles activation functions, convolutional layers, pooling layers, and dense layers
        automatically.
        These can be set manually as well.

    Note:
        Layer sizes are automatically calculated with a dataloader and the layer sizes.

    Note:
        CNNCore is built with modular additional settings.
        It's built to be easily mutable without significant changes.
    """
    # activation list
    _allowed_acts: List[str] = [
        'ReLU',
        'Softplus',
        'Softmax',
        'Tanh',
        'Sigmoid',
        'Mish'
    ]

    def __init__(self, *, ikwiad: bool = False):
        r"""
        **Instantiates CNNCore.**

        Args:
            ikwiad (bool), default = False: Turns off all warning messages ("I know what I am doing" - ikwiad).
        """
        super(CNNCore, self).__init__()

        # activation list
        self._allowed_acts: List[str] = self.allowed_acts()

        # internals
        self.forward: Callable = self.forward
        self.forward_no_grad: Callable = self.forward_no_grad
        # internal checkers
        self._ikwiad: bool = bool(ikwiad)
        self._instantiations: Dict[str, bool] = {
            'channels': False,
            'training_params': False,
            'activators': False,
            'convolutional': False,
            'pooling': False,
            'dense': False,
            'fully_instantiated': False
        }
        # network features
        self._in_dims: Optional[Tuple[int, int]] = None
        self._conv_sizes: Optional[List[int]] = None
        self._dense_sizes: Optional[List[int]] = None
        self._conv_params = None
        self._pool_params = None
        self._dense_params = None
        self._act_params = None
        # activation container
        self._conv_acts: nn.ModuleList = nn.ModuleList()
        self._dense_acts: nn.ModuleList = nn.ModuleList()
        # layer containers
        self._conv: nn.ModuleList = nn.ModuleList()
        self._pool: nn.ModuleList = nn.ModuleList()
        self._dense: nn.ModuleList = nn.ModuleList()

    @classmethod
    def allowed_acts(cls):
        return cls._allowed_acts

    @property
    def instantiations(self):
        return self._instantiations

    def set_channels(self, *, conv_channels: Optional[list] = None, dense_channels: Optional[list] = None) -> None:
        r"""
        **Sets the channel sizes for convolutional and dense layers.**

        Internal items are temporarily set to a string item that will be calculated or set and substituted later.

        Args:
            conv_channels (list, optional), default = ['colors', 16, 32, 64, 128]: Convolutional channel sizes.
            dense_channels (list, optional), default = ['flattened', 256, 128, 64, 32]: Dense channel sizes.

        Returns:
            None

        Raises:
            AlreadySetError: If the channels have already been set.
            ValueError: If invalid values were passed for either of the channels.
        """
        # check for duplicate initialization attempts
        if self._instantiations['channels']:
            raise AlreadySetError("Attempted second set of channels.")

        # check channel types
        if not (
            (isinstance(conv_channels, list) and all(isinstance(itm, int) and 0 < itm for itm in conv_channels))
            or conv_channels is None
        ):
            raise ValueError("Attempted convolutional channel set with incorrect type (list of ints or None)")
        if not (
            (isinstance(dense_channels, list) and all(isinstance(itm, int) and 0 < itm for itm in dense_channels))
            or dense_channels is None
        ):
            raise ValueError("Attempted dense channel set with incorrect type (list of ints or None)")

        # set channels
        conv_channels = conv_channels or [16, 32, 64, 128]
        dense_channels = dense_channels or [256, 128, 64, 32]
        # set internal channels
        self._conv_sizes = ['colors', *conv_channels]
        self._dense_sizes = ['flattened', *dense_channels, 'classes']
        self._instantiations['channels'] = True
        return None

    def transfer_training_params(
            self, color_channels: Optional[int] = None,
            classes: Optional[int] = None,
            initial_dims: Optional[tuple] = None,
            *,
            loader: Optional[DataLoader] = None
    ) -> None:
        r"""
        **Transfers DataLoader parameters into the model so internal components can be set.**

        Args:
            color_channels (int, optional):
                Number of input color channels.
            classes (int, optional):
                Number of output classes.
            initial_dims (tuple, optional):
                Dimensions of the input images (height, width).
            loader (DataLoader, optional):
                DataLoader object for directly transferring parameters.

        Returns:
            None

        Raises:
            MissingMethodError: If channels weren't set.
            AlreadySetError: If training parameters were already set.
            ValueError: If invalid values were passed for any of the parameters.
        """
        # check for channel set
        if not self._instantiations['channels']:
            raise MissingMethodError("Channels weren't set")
        # check for duplicate initialization attempts
        if self._instantiations['training_params']:
            raise AlreadySetError("Training parameters can't be set twice")

        if isinstance(loader, DataLoader):
            # transfer from the dataloader
            images, labels = next(iter(loader))
            _, self._conv_sizes[0], h, w = images.shape
            _, self._dense_sizes[-1] = labels.shape
            self._in_dims = (h, w)
        elif loader:
            # loader provided, but not as the correct object
            if not self._ikwiad:
                print()
                warn("A loader was set, but wasn't a torch dataloader and was ignored", UserWarning)
        if not isinstance(loader, DataLoader):
            # check for errors
            if not (isinstance(color_channels, int) and 0 < color_channels,):
                raise ValueError("'color_channels' must be a positive integer")
            if not (isinstance(classes, int) and 0 < classes):
                raise ValueError("'classes' must be a positive integer")
            if not (
                isinstance(initial_dims, tuple)
                and all([isinstance(itm, int) and itm for itm in initial_dims])
                and len(initial_dims) == 2
            ):
                raise ValueError("'initial_dims' must be a tuple of two positive integers")
            # manually transfer dataloader parameters
            self._conv_sizes[0] = color_channels
            self._dense_sizes[-1] = classes
            self._in_dims = initial_dims

        self._instantiations['training_params'] = True
        return None

    def set_acts(self, *, methods: Optional[list] = None, parameters: Optional[list] = None) -> None:
        r"""
        Sets the activation function for all layers.

        Args:
            methods (list, optional):
                List of activation function names for each layer.
                Defaults to ReLU for all layers except the last one, which defaults to Softmax.
            parameters (list, optional):
                List of parameter dictionaries for each activation function.
                Defaults to PyTorch's default parameters.

        Returns:
            None

        Raises:
            MissingMethodError: If channels weren't set.
            AlreadySetError: If activations were already set.
            TypeError: If any methods or parameters were of the wrong type.
            ValueError: If invalid values were passed for any of the methods or parameters.
        """
        # activation parameter reference
        act_params = {
            'ReLU': Params(
                default={'inplace': False},
                dtypes={'inplace': (bool, int)},
                vtypes={'inplace': lambda x: True},
                ctypes={'inplace': lambda x: bool(x)}
            ),
            'Softplus': Params(
                default={
                    'beta': 1.0,
                    'threshold': 20.0
                },
                dtypes={
                    'beta': (float, int),
                    'threshold': (float, int)
                },
                vtypes={
                    'beta': lambda x: 0 < x,
                    'threshold': lambda x: 0 < x
                },
                ctypes={
                    'beta': lambda x: float(x),
                    'threshold': lambda x: float(x)
                }
            ),
            'Softmax': Params(
                default={'dim': None},
                dtypes={'dim': (type(None), int)},
                vtypes={'dim': lambda x: True},
                ctypes={'dim': lambda x: x}
            ),
            'Tanh': Params(
            ),
            'Sigmoid': Params(
            ),
            'Mish': Params(
                default={'inplace': False},
                dtypes={'inplace': (bool, int)},
                vtypes={'inplace': lambda x: True},
                ctypes={'inplace': lambda x: bool(x)}
            )
        }

        # check for channel set
        if not self._instantiations['channels']:
            raise MissingMethodError("Channels weren't set")
        # check for duplicate initialization attempts
        if self._instantiations['activators']:
            raise AlreadySetError("Activators can't be set twice")

        if methods is None:
            # set default activators
            methods = ['ReLU'] * (len(self._dense_sizes) + len(self._conv_sizes) - 3)
            methods.append('Softmax')
            parameters = [{}] * (len(self._dense_sizes) + len(self._conv_sizes) - 2)
        else:
            # check for errors
            if not (len(methods) == len(parameters) == (len(self._conv_sizes) + len(self._dense_sizes) - 2)):
                raise ValueError(
                    "Invalid matching of 'params', 'methods', and channels\n"
                    f"({len(methods)} != {len(parameters)} != {len(self._conv_sizes) + len(self._dense_sizes) - 2})"
                )
            if not all([mth in self._allowed_acts for mth in methods]):
                raise ValueError(
                    f"Invalid methods detected in {methods}\n"
                    f"Choose from: {self._allowed_acts}"
                )
            if not isinstance(methods, list):
                raise TypeError("'methods' must be a list")
            if not isinstance(parameters, list):
                raise TypeError("'parameters' must be a list")

        self._act_params = []
        for mthd, prms in zip(methods, parameters):
            # set activation objects
            act_checker = ParamChecker(
                prefix=f'Activator parameters ({mthd})',
                parameters=act_params[mthd],
                ikwiad=self._ikwiad
            )
            self._act_params.append({
                'mthd': mthd,
                'prms': act_checker(prms)
            })

        self._instantiations['activators'] = True
        return None

    def set_conv(self, *, parameters: Optional[list] = None) -> None:
        r"""
        Sets the convolutional layers.

        Args:
            parameters (list, optional):
                List of parameter dictionaries for convolutional layers.
                Defaults to default PyTorch parameters.

        Returns:
            None

        Raises:
            MissingMethodError: If channels weren't set.
            AlreadySetError: If convolutional layers were already set.
            TypeError: If any parameters were of the wrong type.
            ValueError: If invalid values were passed for any of the parameters.
        """
        # instantiate parameter checker
        conv_params = Params(
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
                'kernel_size': lambda x: (
                        (isinstance(x, int) and 0 < x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
                'stride': lambda x: (
                        (isinstance(x, int) and 0 < x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
                'padding': lambda x: (
                        (isinstance(x, int) and 0 <= x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
                'dilation': lambda x: (
                        (isinstance(x, int) and 0 < x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
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
        conv_checker = ParamChecker(prefix='Convolutional Parameters', parameters=conv_params, ikwiad=self._ikwiad)

        # check for channel set
        if not self._instantiations['channels']:
            raise MissingMethodError("Channels weren't set")
        # check for duplicate initialization attempts
        if self._instantiations['convolutional']:
            raise AlreadySetError("Convolutional layers can't be set twice")

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._conv_sizes) - 1)
        else:
            # check parameter list
            if not isinstance(parameters, list):
                raise TypeError("'parameters' must be a list")
            if len(parameters) != len(self._conv_sizes) - 1:
                raise ValueError(
                    "'parameters' length must match conv layers\n"
                    f"({len(parameters)} != {len(self._conv_sizes) - 1})"
                )

        # validate conv params
        self._conv_params = [conv_checker(prm) for prm in parameters]
        self._instantiations['convolutional'] = True
        return None

    def set_pool(self, *, parameters: Optional[list] = None) -> None:
        r"""
            Sets the pooling layers.

            Args:
                parameters (list, optional):
                    List of parameter dictionaries for convolutional layers.
                    Defaults to default PyTorch parameters.

            Returns:
                None

            Raises:
                MissingMethodError: If channels weren't set.
                AlreadySetError: If pooling layers were already set.
                TypeError: If any parameters were of the wrong type.
                ValueError: If invalid values were passed for any of the parameters.
            """
        # instantiate parameter checker
        pool_params = Params(
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
                'kernel_size': lambda x: (
                        (isinstance(x, int) and 0 < x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
                'stride': lambda x: x is None or (
                        (isinstance(x, int) and 0 < x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
                'padding': lambda x: (
                        (isinstance(x, int) and 0 <= x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
                'dilation': lambda x: (
                        (isinstance(x, int) and 0 < x)
                        or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) and 0 < i for i in x))
                ),
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
        pool_checker = ParamChecker(prefix='Pooling Parameters', parameters=pool_params, ikwiad=self._ikwiad)

        # check for channel set
        if not self._instantiations['channels']:
            raise MissingMethodError("Channels weren't set")
        # check for duplicate initialization attempts
        if self._instantiations['pooling']:
            raise AlreadySetError("Pooling layers can't be set twice")

        if parameters is None:
            # form parameter list
            parameters = [None] * (len(self._conv_sizes) - 1)
        else:
            # check parameter list
            if not isinstance(parameters, list):
                raise TypeError("'parameters' must be a list")
            if len(parameters) != len(self._conv_sizes) - 1:
                raise ValueError(
                    "'parameters' length must match conv layers\n"
                    f"({len(parameters)} != {len(self._conv_sizes) - 1})"
                )

        # validate pool params
        self._pool_params = [pool_checker(prm) if prm is not None else None for prm in parameters]
        self._instantiations['pooling'] = True
        return None

    def set_dense(self, *, parameters: Optional[list] = None) -> None:
        r"""
        Sets the dense layers.

        Args:
            parameters (list, optional):
                List of parameter dictionaries for dense layers.
                Defaults to default PyTorch parameters.

        Returns:
            None

        Raises:
            MissingMethodError: If channels weren't set.
            AlreadySetError: If dense layers were already set.
            TypeError: If any parameters were of the wrong type.
            ValueError: If invalid values were passed for any of the parameters.
        """
        # initialize parameter checker
        dense_params = Params(
            default={'bias': True},
            dtypes={'bias': (bool, int)},
            vtypes={'bias': lambda x: True},
            ctypes={'bias': lambda x: bool(x)}
        )
        dense_checker = ParamChecker(prefix='Dense Parameters', parameters=dense_params, ikwiad=self._ikwiad)

        # check for channel set
        if not self._instantiations['channels']:
            raise MissingMethodError("Channels weren't set")
        # check for duplicate initialization attempts
        if self._instantiations['dense']:
            raise AlreadySetError("Dense layers can't be set twice")

        if parameters is None:
            # form parameter list
            parameters = [{}] * (len(self._dense_sizes) - 1)
        else:
            # check parameter list
            if not isinstance(parameters, list):
                raise TypeError("'parameters' must be a list")
            if len(parameters) != len(self._dense_sizes) - 1:
                raise ValueError(
                    "'parameters' length must match dense layers\n"
                    f"({len(parameters)} != {len(self._dense_sizes) - 1})"
                )

        # validate dense params
        self._dense_params = [dense_checker(prm) for prm in parameters]
        self._instantiations['dense'] = True
        return None

    @staticmethod
    def _calc_lyr_size(dims: tuple, params: dict):
        # parameter reformatting
        params = {key: (val, val) if isinstance(val, int) else val for key, val in params.items()}
        # out size calculation
        h_in, w_in = dims
        h_out = (
                (h_in + 2 * params['padding'][0] - params['dilation'][0] * (params['kernel_size'][0] - 1) - 1)
                // params['stride'][0]
                + 1
        )
        w_out = (
                (w_in + 2 * params['padding'][1] - params['dilation'][1] * (params['kernel_size'][1] - 1) - 1)
                // params['stride'][1]
                + 1
        )
        # return out size
        return h_out, w_out

    def instantiate_model(self, *, crossentropy: bool = True) -> None:
        r"""
        Builds the full model with PyTorch and allows the model to be used.

        Args:
            crossentropy (bool, optional):
                If crossentropy loss is used in the model.
                Used for internal instantiation.

        Returns:
            None

        Raises:
            MissingMethodError: If all the necessary components weren't set.
        """
        # check for proper instantiation
        nec_instantiations = self._instantiations.copy()
        nec_instantiations.pop('fully_instantiated')
        if not all(nec_instantiations.values()):
            raise MissingMethodError(
                "Necessary settings weren't fully instantiated:\n"
                f"{nec_instantiations}"
            )

        dims = self._in_dims
        for conv, pool in zip(self._conv_params, self._pool_params):
            # grab necessary parameters
            calc_conv_params = {
                'padding': conv['padding'],
                'dilation': conv['dilation'],
                'kernel_size': conv['kernel_size'],
                'stride': conv['stride']
            }
            dims = self._calc_lyr_size(dims, calc_conv_params)
            if pool is not None:
                calc_pool_params = {
                    'padding': pool['padding'],
                    'dilation': pool['dilation'],
                    'kernel_size': pool['kernel_size'],
                    'stride': pool['kernel_size'] if pool['stride'] is None else pool['stride']
                }
                dims = self._calc_lyr_size(dims, calc_pool_params)

        # set flattened size
        h, w = dims
        final_size = int(h * w * self._conv_sizes[-1])
        if final_size <= 0 and not self._ikwiad:
            print()
            warn(
                "This model is about to be instantiated with a flattened size of 0",
                UserWarning
            )
        self._dense_sizes[0] = final_size

        for i, (conv, pool) in enumerate(zip(self._conv_params, self._pool_params)):
            # set conv and pool layers
            self._conv.append(nn.Conv2d(*self._conv_sizes[i:i + 2], **conv))
            if pool is not None:
                self._pool.append(nn.MaxPool2d(**pool))
            else:
                self._pool.append(nn.Identity())
        for i, prms in enumerate(self._dense_params):
            # set dense layers
            self._dense.append(nn.Linear(*self._dense_sizes[i:i + 2], **prms))

        # set forward method
        self.forward, self.forward_no_grad = self._compile_forward(bool(crossentropy))
        self._instantiations['fully_instantiated'] = True
        return None

    def _compile_forward(self, crossentropy):
        # torch activation reference
        activation_ref = {
            'ReLU': nn.ReLU,
            'Softplus': nn.Softplus,
            'Softmax': nn.Softmax,
            'Tanh': nn.Tanh,
            'Sigmoid': nn.Sigmoid,
            'Mish': nn.Mish
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
                x = torch.flatten(x, 0)
                for act, dense in zip(self._dense_acts[:-1], self._dense[:-1]):
                    x = act(dense(x))
                return self._dense[-1](x)

        def _forward_no_grad(x: torch.Tensor) -> torch.Tensor:
            # set forward w/o grad
            with torch.no_grad():
                for act, conv, pool in zip(self._conv_acts, self._conv, self._pool):
                    x = pool(act(conv(x)))
                x = torch.flatten(x, 0)
                for act, dense in zip(self._dense_acts[:-1], self._dense[:-1]):
                    x = act(dense(x))
                return self._dense[-1](x)

        return _forward, _forward_no_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Runs a forward pass of the model if the model is fully built.
        instantiate_model must be run before this function can be run.

        Args:
            x (torch.Tensor):
                Inputs to the model.

        Returns:
            torch.Tensor:
                Outputs to the model.

        Raises:
            MissingMethodError: If the model hasn't been fully instantiated.
        """
        # model not fully instantiated
        raise MissingMethodError(
            "Model wasn't fully instantiated\n"
            f"({self._instantiations})"
        )

    def forward_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Runs a forward pass of the model if the model is fully built without grad
        and with the final activation method.
        instantiate_model must be run before this function can be run.

        Args:
            x (torch.Tensor):
                Inputs to the model.

        Returns:
            torch.Tensor:
                Outputs to the model.

        Raises:
            MissingMethodError: If the model hasn't been fully instantiated.
        """
        # model not fully instantiated
        raise MissingMethodError(
            "Model wasn't fully instantiated\n"
            f"({self._instantiations})"
        )
