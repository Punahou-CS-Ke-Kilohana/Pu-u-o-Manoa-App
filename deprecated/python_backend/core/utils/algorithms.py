r"""
This module consists of algorithms for the Pu-u-o-Manoa-App.
It contains an optimizer setter and loss setter.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""
# todo: clean up some code and add docstrings to this file

from typing import Iterator, Optional, Union
import torch.optim
import torch.optim as optim
import torch.nn as nn

from .utils import ParamChecker


class OptimSetter:
    r"""
    OptimSetter is the optimizer setter for a PyTorch model.

    This class allows for the ease of setting and verification of an optimizer. It's mainly
    used due to its ease of integration with configs. Any internal components that are
    restricted, such as activations, can be altered with relative ease.
    """

    _allowed_optims = [
        'Adagrad',
        'Adam',
        'AdamW',
        'LBFGS',
        'RMSprop',
        'SGD'
    ]

    def __init__(self, *, ikwiad: bool = False):
        r"""
        Initializes the OptimSetter class.

        Args:
            ikwiad (bool, optional):
                "I know what I am doing" (ikwiad).
                If True, removes all warning messages.
                Defaults to False.
        """
        # allowed optim list
        self._allowed_optims = self.allowed_optims

        # internals
        self._ikwiad = bool(ikwiad)
        self._method = None
        self._hyperparameters = None

    @classmethod
    def allowed_optims(cls):
        return cls._allowed_optims

    def set_hyperparameters(
            self,
            method: Optional[str] = None,
            *,
            hyperparameters: Optional[dict] = None,
            **kwargs
    ) -> None:
        r"""
        ...

        Args:
            method:
            hyperparameters:
            **kwargs:

        Returns:
            None

        Raises:
            ...
        """
        # optim hyperparameter reference
        optim_hyperparams = {
            'Adagrad': {
                'default': {
                    'lr': 0.01,
                    'lr_decay': 0.0,
                    'weight_decay': 0.0,
                    'initial_accumulator_value': 0.0,
                    'eps': 1e-10,
                    'foreach': None,
                    'maximize': False,
                    'differentiable': False,
                    'fused': None
                },
                'dtypes': {
                    'lr': (float, int),
                    'lr_decay': (float, int),
                    'weight_decay': (float, int),
                    'initial_accumulator_value': (float, int),
                    'eps': float,
                    'foreach': (type(None), bool, int),
                    'maximize': (bool, int),
                    'differentiable': (bool, int),
                    'fused': (type(None), bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0 < x,
                    'lr_decay': lambda x: 0 < x,
                    'weight_decay': lambda x: 0 < x,
                    'initial_accumulator_value': lambda x: 0 < x,
                    'eps': lambda x: 0 < x < 1,
                    'foreach': lambda x: True,
                    'maximize': lambda x: True,
                    'differentiable': lambda x: True,
                    'fused': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'lr_decay': lambda x: float(x),
                    'weight_decay': lambda x: float(x),
                    'initial_accumulator_value': lambda x: float(x),
                    'eps': lambda x: x,
                    'foreach': lambda x: bool(x) if x is not None else None,
                    'maximize': lambda x: bool(x),
                    'differentiable': lambda x: bool(x),
                    'fused': lambda x: bool(x) if x is not None else None
                }
            },
            'Adam': {
                'default': {
                    'lr': 0.01,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0.0,
                    'amsgrad': False,
                    'foreach': None,
                    'maximize': False,
                    'capturable': False,
                    'differentiable': False,
                    'fused': None
                },
                'dtypes': {
                    'lr': (float, int),
                    'betas': tuple,
                    'eps': float,
                    'weight_decay': (float, int),
                    'amsgrad': (bool, int),
                    'foreach': (type(None), bool, int),
                    'maximize': (bool, int),
                    'capturable': (bool, int),
                    'differentiable': (bool, int),
                    'fused': (type(None), bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0 < x,
                    'betas': lambda x: len(x) == 2 and (isinstance(itm, float) and 0 < itm < 1 for itm in x),
                    'eps': lambda x: 0 < x < 1,
                    'weight_decay': lambda x: 0 < x,
                    'amsgrad': lambda x: True,
                    'foreach': lambda x: True,
                    'maximize': lambda x: True,
                    'capturable': lambda x: True,
                    'differentiable': lambda x: True,
                    'fused': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'betas': lambda x: x,
                    'eps': lambda x: x,
                    'weight_decay': lambda x: float(x),
                    'amsgrad': lambda x: bool(x),
                    'foreach': lambda x: bool(x) if x is not None else None,
                    'maximize': lambda x: bool(x),
                    'capturable': lambda x: bool(x),
                    'differentiable': lambda x: bool(x),
                    'fused': lambda x: bool(x) if x is not None else None
                }
            },
            'AdamW': {
                'default': {
                    'lr': 0.01,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0.0,
                    'amsgrad': False,
                    'maximize': False,
                    'foreach': None,
                    'capturable': False,
                    'differentiable': False,
                    'fused': None
                },
                'dtypes': {
                    'lr': (float, int),
                    'betas': tuple,
                    'eps': float,
                    'weight_decay': (float, int),
                    'amsgrad': (bool, int),
                    'maximize': (bool, int),
                    'foreach': (type(None), bool, int),
                    'capturable': (bool, int),
                    'differentiable': (bool, int),
                    'fused': (type(None), bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0 < x,
                    'betas': lambda x: len(x) == 2 and (isinstance(itm, float) and 0 < itm < 1 for itm in x),
                    'eps': lambda x: 0 < x < 1,
                    'weight_decay': lambda x: 0 < x,
                    'amsgrad': lambda x: True,
                    'maximize': lambda x: True,
                    'foreach': lambda x: True,
                    'capturable': lambda x: True,
                    'differentiable': lambda x: True,
                    'fused': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'betas': lambda x: x,
                    'eps': lambda x: x,
                    'weight_decay': lambda x: float(x),
                    'amsgrad': lambda x: bool(x),
                    'maximize': lambda x: bool(x),
                    'foreach': lambda x: bool(x) if x is not None else None,
                    'capturable': lambda x: bool(x),
                    'differentiable': lambda x: bool(x),
                    'fused': lambda x: bool(x) if x is not None else None
                }
            },
            'RMSprop': {
                'default': {
                    'lr': 0.01,
                    'alpha': 0.99,
                    'eps': 1e-08,
                    'weight_decay': 0.0,
                    'momentum': 0.0,
                    'centered': False,
                    'capturable': False,
                    'foreach': None,
                    'maximize': False,
                    'differentiable': False
                },
                'dtypes': {
                    'lr': (float, int),
                    'alpha': float,
                    'eps': float,
                    'weight_decay': (float, int),
                    'momentum': float,
                    'centered': (bool, int),
                    'capturable': (bool, int),
                    'foreach': (type(None), bool, int),
                    'maximize': (bool, int),
                    'differentiable': (bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0 < x,
                    'alpha': lambda x: 0 < x < 1,
                    'eps': lambda x: 0 < x < 1,
                    'weight_decay': lambda x: 0 < x,
                    'momentum': lambda x: 0 < x < 1,
                    'centered': lambda x: True,
                    'capturable': lambda x: True,
                    'foreach': lambda x: True,
                    'maximize': lambda x: True,
                    'differentiable': lambda x: True,
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'alpha': lambda x: x,
                    'eps': lambda x: x,
                    'weight_decay': lambda x: float(x),
                    'momentum': lambda x: x,
                    'centered': lambda x: bool(x),
                    'capturable': lambda x: bool(x),
                    'foreach': lambda x: bool(x) if x is not None else None,
                    'maximize': lambda x: bool(x),
                    'differentiable': lambda x: bool(x),
                }
            },
            'SGD': {
                'default': {
                    'lr': 0.01,
                    'momentum': 0.0,
                    'dampening': 0.0,
                    'weight_decay': 0.0,
                    'nesterov': False,
                    'maximize': False,
                    'foreach': None,
                    'differentiable': False,
                    'fused': None
                },
                'dtypes': {
                    'lr': (float, int),
                    'momentum': float,
                    'dampening': float,
                    'weight_decay': (float, int),
                    'nesterov': (bool, int),
                    'maximize': (bool, int),
                    'foreach': (type(None), bool, int),
                    'differentiable': (bool, int),
                    'fused': (type(None), bool, int)
                },
                'vtypes': {
                    'lr': lambda x: 0 < x,
                    'momentum': lambda x: 0 < x < 1,
                    'dampening': lambda x: 0 < x < 1,
                    'weight_decay': lambda x: 0 < x,
                    'nesterov': lambda x: True,
                    'maximize': lambda x: True,
                    'foreach': lambda x: True,
                    'differentiable': lambda x: True,
                    'fused': lambda x: True
                },
                'ctypes': {
                    'lr': lambda x: float(x),
                    'momentum': lambda x: x,
                    'dampening': lambda x: x,
                    'weight_decay': lambda x: float(x),
                    'nesterov': lambda x: bool(x),
                    'maximize': lambda x: bool(x),
                    'foreach': lambda x: bool(x) if x is not None else None,
                    'differentiable': lambda x: bool(x),
                    'fused': lambda x: bool(x) if x is not None else None
                }
            }
        }

        # check types and methods
        if not isinstance(method, str):
            raise TypeError("'method' must be a string")
        if method not in self._allowed_optims:
            raise ValueError(
                f"Invalid method: {method}\n"
                f"Choose from: {self._allowed_optims}"
            )

        # check hyperparameters
        optim_checker = ParamChecker(name=f'{method} Parameters', ikwiad=self._ikwiad)
        optim_checker.set_types(**optim_hyperparams[method])

        # set internal hyperparameters
        self._method = method
        self._hyperparameters = optim_checker.check_params(hyperparameters, **kwargs)
        return None

    def get_optim(self, parameters: Union[Iterator[nn.Parameter], nn.Parameter]) -> optim.Optimizer:
        # torch optim reference
        optim_ref = {
            'Adagrad': optim.Adagrad,
            'Adam': optim.Adam,
            'AdamW': optim.AdamW,
            'LBFGS': optim.LBFGS,
            'RMSprop': optim.RMSprop,
            'SGD': optim.SGD
        }
        # return optim object
        return optim_ref[self._method](parameters, **self._hyperparameters)


class LossSetter:
    _allowed_losses = [
        'CrossEntropyLoss',
        'MSELoss',
        'L1Loss'
    ]

    def __init__(self, *, ikwiad: bool = False):
        # allowed loss list
        self._allowed_losses = self.allowed_losses

        # internals
        self._ikwiad = bool(ikwiad)
        self._method = None
        self._hyperparameters = None

    @classmethod
    def allowed_losses(cls):
        return cls._allowed_losses

    def set_hyperparameters(
            self,
            method: Optional[str] = None,
            *,
            hyperparameters: Optional[dict] = None,
            **kwargs
    ) -> None:
        # loss hyperparameter reference
        loss_hyperparams = {
            'CrossEntropyLoss': {
                'default': {
                    'weight': None,
                    'size_average': None,
                    'ignore_index': -100,
                    'reduce': None,
                    'reduction': 'mean',
                    'label_smoothing': 0.0
                },
                'dtypes': {
                    'weight': torch.Tensor,
                    'size_average': (type(None), bool, int),
                    'ignore_index': int,
                    'reduce': (type(None), bool, int),
                    'reduction': str,
                    'label_smoothing': (float, int)
                },
                'vtypes': {
                    'weight': lambda x: True,
                    'size_average': lambda x: True,
                    'ignore_index': lambda x: True,
                    'reduce': lambda x: True,
                    'reduction': lambda x: x in ['none', 'mean', 'sum'],
                    'label_smoothing': lambda x: 0 <= x <= 1
                },
                'ctypes': {
                    'weight:': lambda x: x,
                    'size_average': lambda x: bool(x) if x is not None else None,
                    'ignore_index': lambda x: x,
                    'reduce': lambda x: bool(x) if x is not None else None,
                    'reduction': lambda x: x,
                    'label_smoothing': lambda x: float(x)
                }
            },
            'MSELoss': {
                'default': {
                    'size_average': None,
                    'reduce': None,
                    'reduction': 'mean'
                },
                'dtypes': {
                    'size_average': (type(None), bool, int),
                    'reduce': (type(None), bool, int),
                    'reduction': str
                },
                'vtypes': {
                    'size_average': lambda x: True,
                    'reduce': lambda x: True,
                    'reduction': lambda x: x in ['none', 'mean', 'sum']
                },
                'ctypes': {
                    'size_average': lambda x: bool(x) if x is not None else None,
                    'reduce': lambda x: bool(x) if x is not None else None,
                    'reduction': lambda x: x
                }
            },
            'L1Loss': {
                'default': {
                    'size_average': None,
                    'reduce': None,
                    'reduction': 'mean'
                },
                'dtypes': {
                    'size_average': (type(None), bool, int),
                    'reduce': (type(None), bool, int),
                    'reduction': str
                },
                'vtypes': {
                    'size_average': lambda x: True,
                    'reduce': lambda x: True,
                    'reduction': lambda x: x in ['none', 'mean', 'sum']
                },
                'ctypes': {
                    'size_average': lambda x: bool(x) if x is not None else None,
                    'reduce': lambda x: bool(x) if x is not None else None,
                    'reduction': lambda x: x
                }
            }
        }

        # check types and methods
        assert isinstance(method, str),\
            "'method' must be a string"
        assert method in self._allowed_losses, \
            (f"Invalid method: {method}\n"
             f"Choose from: {self._allowed_losses}")

        # check hyperparameters
        loss_checker = ParamChecker(name=f'{method} Parameters', ikwiad=self._ikwiad)
        loss_checker.set_types(**loss_hyperparams[method])

        # set internal hyperparameters
        self._method = method
        self._hyperparameters = loss_checker.check_params(hyperparameters, **kwargs)
        return None

    def get_loss(self) -> nn.Module:
        # torch loss reference
        loss_ref = {
            'CrossEntropyLoss': nn.CrossEntropyLoss,
            'MSELoss': nn.MSELoss,
            'L1Loss': nn.L1Loss
        }
        # return loss object
        return loss_ref[self._method](**self._hyperparameters)
