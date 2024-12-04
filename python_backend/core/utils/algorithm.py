from typing import Optional

import torch.optim
import torch.optim as optim
import torch.nn as nn

from .utils import ParamChecker


class OptimSetter:
    _allowed_optims = [
        'Adagrad',
        'Adam',
        'AdamW',
        'LBFGS',
        'RMSprop',
        'SGD'
    ]

    def __init__(self, *, ikwiad: bool = False):
        self._ikwiad = bool(ikwiad)
        self._allowed_optims = self.get_allowed_optims
        self._method = None
        self._hyperparameters = None

    @classmethod
    def get_allowed_optims(cls):
        return cls._allowed_optims

    def set_hyperparameters(self, method: Optional[str] = None, *, parameters: Optional[dict] = None, **kwargs) -> None:
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
        assert isinstance(method, str), "'method' must be a string"
        assert method in self._allowed_optims, f"Invalid method: {method}\nChoose from: {self._allowed_optims}"

        # check hyperparameters
        optim_checker = ParamChecker(name=f'{method} Parameters', ikwiad=self._ikwiad)
        optim_checker.set_types(**optim_hyperparams[method])

        # set internal hyperparameters
        self._method = method
        self._hyperparameters = optim_checker.check_params(parameters, **kwargs)

    def set_parameters(self, parameters: torch.nn.Parameter) -> torch.optim.Optimizer:
        # torch optim reference
        optim_ref = {
            'Adagrad': torch.optim.Adagrad,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'LBFGS': torch.optim.LBFGS,
            'RMSprop': torch.optim.RMSprop,
            'SGD': torch.optim.SGD
        }
        # return optim object
        return optim_ref[self._method](parameters, **self._hyperparameters)


class LossSetter:
    ...
