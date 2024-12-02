r"""
**Utility functions.**

Attributes:
----------
**ParamChecker**:
    Checks parameters.
"""

import types
import warnings


class ParamChecker:
    r"""
    **Parameter checker for class methods.**

    Checks parameters for a given type based on settings.

    Attributes:
    ----------
    **default** : (*dict*)
        Default parameters.
    **dtypes** : (*dict*)
        Datatypes.
    **vtypes** : (*dict*)
        Value types.
    **ctypes** : (*dict*)
        Conversion types.

    Methods:
    ----------
    **__init__(self, name: str = 'parameters', ikwiad: bool = False)** :
        Instantiates object with given internal settings.

    **set_types(self, default: dict, dtypes: dict, vtypes: dict, ctypes: dict) -> None** :
        Set parameter checker's settings.

    **check_params(self, params: dict, kwargs: dict) -> dict** :
        Checks parameters with parameter checker's settings

    Notes:
    ----------
    - Used internally and externally to check parameters.
    """
    def __init__(self, name: str = 'parameters', ikwiad: bool = False):
        r"""
        **Parameter checker initialization defined settings.**

        Parameters:
        ----------
        **name** : (*str*, *optional*), default='parameters'
            Internal name for debugging purposes.

        **ikwiad** : (*bool*, *optional*), default=False
            Remove weak warning messages (I know what I am doing).

        Notes:
        ----------
        - 'ikwiad' only removes weak warnings; it does not allow bypassing of hard errors.
        - The name parameter lets you know where an error occured and isn't used otherwise.

        Example:
        -----
        >>> checker = ParamChecker(name='kernel checker')
        """
        # warning settings
        self._name = str(name)
        self._ikwiad = bool(ikwiad)

        # bool settings
        self._is_set = False

        # internal default settings
        self._default = None
        self._dtypes = None
        self._vtypes = None
        self._ctypes = None

    def set_types(self, default: dict = None, dtypes: dict = None, vtypes: dict = None, ctypes: dict = None) -> None:
        r"""
        **Default class settings.**

        Parameters:
        ----------
        **default** (*dict*) :
            Default parameters, which cannot be callable.

        **dtypes** (*dict*) :
            Default datatypes, which cannot be callable.

        **vtypes** (*dict*) :
            Default value types, which must be lambda functions that return a bool.

        **ctypes** (*dict*) :
            Default conversion types, which must be a lambda function that converts values.

        Notes:
        ----------
        - Setting lambda functions to return True for vtypes will bypass any value conditionals.

        - Setting lambda functions to return x for ctypes will bypass any conversions.

        - If no default parameters are given, dtypes, vtypes, and ctypes will not be neccesary.
            - There is also no reason to use this class if this is the case.

        Example:
        -----
        >>> checker = ParamChecker(name='kernel checker')
        >>> checker.set_types(default={'size': 3}, dtypes={'size': int}, vtypes={'size': lambda x: x > 0}, ctypes={'size': lambda x: x})
        """
        if default is not None or not isinstance(dtypes, dict):
            # set default
            if default is None or all(not callable(value) for value in default.values()):
                # set types
                self._default = default
            else:
                # invalid type
                raise TypeError(f"Invalid datatype for '{self._name}'s default': '{default}'")

        if dtypes is not None or not isinstance(dtypes, dict):
            # set datatypes
            if dtypes is None or all(not callable(value) for value in default.values()):
                # set types
                self._dtypes = dtypes
            else:
                # invalid type
                raise TypeError(f"Invalid datatype for '{self._name}'s dtypes': '{dtypes}'")

        if vtypes is not None or not isinstance(dtypes, dict):
            # set value types
            if vtypes is None or all(isinstance(value, types.LambdaType) for value in vtypes.values()):
                # set types
                self._vtypes = vtypes
            else:
                # invalid type
                raise TypeError(f"Invalid datatype for '{self._name}'s vtypes': '{vtypes}'")

        if ctypes is not None or not isinstance(dtypes, dict):
            # set conversion types
            if ctypes is None or all(isinstance(value, types.LambdaType) for value in ctypes.values()):
                # set types
                self._ctypes = ctypes
            else:
                # invalid type
                raise TypeError(f"Invalid datatype for '{self._name}'s ctypes': '{ctypes}'")

        if default is not None:
            if not default.keys() == dtypes.keys() == vtypes.keys() == ctypes.keys():
                # check keys
                raise ValueError(
                    f"Keys don't match for {self._name}'s parameters\n"
                    f"default: {default.keys()}\n"
                    f"dtypes: {dtypes.keys()}\n"
                    f"vtypes: {vtypes.keys()}\n"
                    f"ctypes: {ctypes.keys()}"
                )

        # default parameters set
        self._is_set = True
        return None

    def check_params(self, params: dict, **kwargs: dict) -> dict:
        r"""
        **Parameter checker application.**

        Parameters:
        ----------
        **params** (*dict*) :
            Fed-in parameters.

        ****kwargs** (*any*, *optional*) :
            The parameters with keyword arguments if desired.

        Notes:
        ----------
        - Any parameters not specified will be set to their default values.

        - Parameters that are specified but not used within the specified algorithm will be discarded.
            - The user will receive a warning when this occurs.

        Returns:
        ----------
        - **params** : *dict*
            The checked parameters.

        Example:
        -----
        >>> checker = ParamChecker(name='kernel checker')
        >>> checker.set_types(default={'size': 3}, dtypes={'size': int}, vtypes={'size': lambda x: x > 0}, ctypes={'size': lambda x: x})
        >>> kernel_params = checker.check_params(size=5)
        """
        if not self._is_set:
            # check if default parameters have been set
            raise RuntimeError(f"Default parameters not set for '{self._name}'")

        # instantiate parameter dictionary
        prms = self._default

        # combine keyword arguments and parameters
        if params and kwargs:
            params.update(kwargs)
        elif kwargs:
            params = kwargs

        if params and (prms is not None) and isinstance(params, dict):
            # set defined parameter
            for prm in params:
                if prm not in prms:
                    # invalid parameter
                    if not self._ikwiad:
                        print()
                        warnings.warn(
                            f"\nInvalid parameter for '{self._name}': '{prm}'\n"
                            f"Choose from: '{[prm for prm in prms]}'",
                            UserWarning
                        )
                elif prm in prms and not isinstance(params[prm], self._dtypes[prm]):
                    # invalid datatype for parameter
                    raise TypeError(
                        f"Invalid datatype for '{self._name}': '{prm}'\n"
                        f"Choose from: {self._dtypes[prm]}"
                    )
                elif prm in prms and not (self._vtypes[prm](params[prm])):
                    # invalid value for parameter
                    raise TypeError(
                        f"Invalid value for '{self._name}': '{prm}'\n"
                        f"Failed conditional: {self._vtypes[prm]}"
                    )
                else:
                    # valid parameter
                    prms[prm] = self._ctypes[prm](params[prm])
        elif params and isinstance(params, dict):
            # parameters not taken
            if not self._ikwiad:
                print()
                warnings.warn(f"\n'{self._name}' does not take parameters", UserWarning)
        elif params:
            # invalid data type
            raise TypeError(
                f"'params' is not a dictionary: {params}\n"
                f"Choose from: {[prm for prm in prms]}"
            )

        # return parameters
        return prms
