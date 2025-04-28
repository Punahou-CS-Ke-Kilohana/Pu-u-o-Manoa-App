r"""
**Setup for library installation.**

Run this file to install the library.

In the library root, run either:
    - pip install .
    - pip3 install .

To install the library using editable mode, run either:
    - pip install -e .
    - pip3 install -e .

Verify installation using either:
    - pip list
    - pip3 list

If recognition failed, run either:
    - mac: export PYTHONPATH=$(pwd)
    - windows: set PYTHONPATH=%cd%
"""

from setuptools import setup, find_packages

setup(
    name='RHCCCore',
    version='0.9.0',
    packages=find_packages()
)
