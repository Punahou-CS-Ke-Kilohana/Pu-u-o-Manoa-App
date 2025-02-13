r"""
This file is the execution script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

from application.scripts.train import train


def main(name: str, *, ikwiad=False) -> None:
    r"""
    The main execution script for the Pu-u-o-Manoa-App.

    Args:
        name (str):
            The method name.
        ikwiad (bool, optional):
            "I know what I am doing" (ikwiad).
            If True, removes all warning messages.
            Defaults to False.

    Returns:
        None
    """
    # method reference
    methods = {
        'train': train,
        'validate': ...,
        'interpret': ...
    }
    method_titles = {
        'train': 'Training',
        'validate': 'Validating',
        'interpret': 'Interpreting'
    }
    if name not in methods.keys():
        # invalid method
        raise ValueError(
            "'name' must refer to a valid method\n"
            f"({list(methods.keys())})"
        )

    # method execution
    print(f"Executing {method_titles[name]}")
    methods[name](ikwiad=ikwiad)
    return None


if __name__ == '__main__':
    import torch.multiprocessing as mp
    from method_selector import method_name

    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main(method_name)
