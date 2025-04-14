r"""
This module consists of the errors for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""


class AlreadySetError(Exception):
    # if something was already set
    pass


class MissingMethodError(Exception):
    # if a method hasn't been run
    pass
