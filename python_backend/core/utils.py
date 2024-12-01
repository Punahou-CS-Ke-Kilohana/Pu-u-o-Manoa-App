from typing import Union
from types import LambdaType
import warnings


class ParamChecker:
    def __init__(self, name: str = 'Parameters', ikwiad: bool = False):
        # warning settings
        self._name = str(name)
        self._ikwiad = bool(ikwiad)

        # instantiation checker
        self._is_set = False

        # internal checkers
        self._default = None
        self._dtypes = None
        self._vtypes = None
        self._ctypes = None

    def _validate_dict(self, param_dict, name, check_lambda=False):
        assert isinstance(param_dict, dict), \
            f"'{name}' must be a dictionary ({type(param_dict)} != dict)"
        if check_lambda:
            assert all(isinstance(value, LambdaType) for value in param_dict.values()), \
                f"Invalid datatype for '{self._name}'s {name}': '{param_dict}'"
        else:
            assert all(not callable(value) for value in param_dict.values()), \
                f"Invalid datatype for '{self._name}'s {name}': '{param_dict}'"

    def set_types(self,
                  default: Union[dict, None] = None,
                  dtypes: Union[dict, None] = None,
                  vtypes: Union[dict, None] = None,
                  ctypes: Union[dict, None] = None) -> None:
        if default is None:
            # default none
            return None

        # validate dicts
        self._validate_dict(default, 'default')
        self._validate_dict(dtypes, 'dtypes')
        self._validate_dict(vtypes, 'vtypes', check_lambda=True)
        self._validate_dict(ctypes, 'ctypes', check_lambda=True)

        # check for key matching
        keys = [default.keys(), dtypes.keys(), vtypes.keys(), ctypes.keys()]
        assert all(k == keys[0] for k in keys), f"Keys don't match for '{self._name}'"

        # set internal checkers
        self._default = default
        self._dtypes = dtypes
        self._vtypes = vtypes
        self._ctypes = ctypes
        self._is_set = True
        return None

    def check_params(self, params: Union[dict, None] = None, **kwargs) -> Union[dict, None]:
        # check for default parameters
        assert self._is_set, f"Default parameters not set for '{self._name}'"

        # initialize as default
        final = self._default.copy()

        # return default
        if params is None and kwargs is None:
            return final

        assert isinstance(params, dict) or kwargs, f"'params' must be a dictionary ({type(params)} != dict)"
        params = params if params else {}
        if kwargs:
            params.update(kwargs)

        for key, prm in params:
            if key not in self._default:
                # invalid key
                if not self._ikwiad:
                    print()
                    warnings.warn(
                        f"\nInvalid parameter for '{self._name}': '{key}'\n"
                        f"Choose from: '{[pos for pos in self._default]}'",
                        UserWarning
                    )
                continue
            # datatype check
            assert isinstance(prm, self._dtypes[key]), \
                (f"Invalid datatype for '{self._name}': '{prm}'\n"
                 f"Choose from: {self._dtypes[prm]}")
            assert self._vtypes[key](prm), \
                (f"Invalid value for '{self._name}': '{prm}'\n"
                 f"Failed conditional: {self._vtypes[prm]}")
            final[key] = self._ctypes[key](prm)

        # return parameters
        return final
