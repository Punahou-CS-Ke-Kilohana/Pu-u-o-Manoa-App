r"""
This module consists of the validating script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import importlib.util
import os
import torch
from typing import Optional

from RHCCCore.network import CNNCore
from application.dataloader.dataloader import loader

from application.configs.training_config import training_config


def set_model(device: torch.device, name: str, epoch: Optional[int], ikwiad: bool) -> CNNCore:
    model = CNNCore(ikwiad=ikwiad)
    try:
        epoch = int(epoch)
    except TypeError or ValueError:
        pass
        # debug
    if epoch is not None:
        assert isinstance(epoch, int)
    assert isinstance(device, torch.device)

    # set model
    model_path = os.path.join(f"{training_config.save_params['save_root']}", f"{name}")
    if not isinstance(epoch, int):
        # take latest epoch
        epoch = max([int(model[6:]) for model in os.listdir(model_path) if model.startswith('epoch_')])

    # import config
    spec = importlib.util.spec_from_file_location('model_config', os.path.join(model_path, 'model_config.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_config = mod.model_config
    # make model
    model.set_channels(conv_channels=model_config.conv.sizes, dense_channels=model_config.dense.sizes)
    model.transfer_training_params(loader=loader)
    model.set_acts(methods=model_config.acts.methods, parameters=model_config.acts.params)
    model.set_conv(parameters=model_config.conv.conv_params)
    model.set_pool(parameters=model_config.conv.pool_params)
    model.set_dense(parameters=model_config.dense.params)
    model.instantiate_model(crossentropy=False)
    # load model
    model.load_state_dict(torch.load(os.path.join(model_path, f'epoch_{epoch}'), weights_only=True))
    model = model.to(device=device)
    model.eval()
    return model


def validate(
        device: torch.device = torch.device('cpu'),
        name: str = None,
        epoch: Optional[int] = None,
        ikwiad: bool = False
) -> None:
    r"""
    Validates a trained model.

    Args:
        device (torch.device):
            Utilized device.
        name (str):
            Name the loaded model.
        epoch (int):
            Loaded epoch.
        ikwiad (bool, optional):
            "I know what I am doing" (ikwiad).
            If True, removes all warning messages.
            Defaults to False.

    Returns:
        None
    """
    # initializations
    ikwiad = bool(ikwiad)
    model = set_model(device=device, name=name, epoch=epoch, ikwiad=ikwiad)
    for itm in loader:
        print(model.forward_no_grad(x=itm))
