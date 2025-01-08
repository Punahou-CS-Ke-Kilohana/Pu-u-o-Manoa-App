r"""
This file is where you select the intended main script for the Pu-u-o-Manoa-App.
It's moved here so those more unfamiliar with this code can easily run the necessary methods.
Run main.py to execute your selected method.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

# select the script method here
method_name = 'train'
possible_methods = ['train', 'validate', 'interpret']

# expose to import
__all__ = ['method_name']
