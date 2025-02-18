r"""
This module consists of the training script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import time
import os
import sys
import shutil
import torch
from typing import Optional
from warnings import warn

from RHCCCore.network import CNNCore
from RHCCCore.utils import (
    OptimSetter,
    LossSetter,
    convert_time,
    progress
)

from application.configs.hyperparameters_config import hyperparameters_config
from application.configs.model_config import model_config
from application.dataloader.dataloader import loader
from application.configs.training_config import training_config


def train(
        device: torch.device = torch.device('cpu'),
        name: Optional[str] = None,
        epoch: Optional[int] = None,
        ikwiad: bool = False
) -> None:
    r"""
    Trains the model using the configs in the config folder.

    Args:
        device (torch.device):
            Utilized device.
        name (str):
            Name the model will save with.
        epoch (int):
            Resumed epoch.
        ikwiad (bool, optional):
            "I know what I am doing" (ikwiad).
            If True, removes all warning messages.
            Defaults to False.

    Returns:
        None
    """
    ikwiad = bool(ikwiad)
    try:
        epoch = int(epoch)
    except TypeError or ValueError:
        pass
    # initialize model
    model = CNNCore(ikwiad=ikwiad)
    model.set_channels(conv_channels=model_config.conv.sizes, dense_channels=model_config.dense.sizes)
    model.transfer_training_params(loader=loader)
    model.set_acts(methods=model_config.acts.methods, parameters=model_config.acts.params)
    model.set_conv(parameters=model_config.conv.conv_params)
    model.set_pool(parameters=model_config.conv.pool_params)
    model.set_dense(parameters=model_config.dense.params)
    model.instantiate_model(crossentropy=(hyperparameters_config.loss.method == 'CrossEntropyLoss'))
    model = model.to(device=device)

    # set loss
    loss_setter = LossSetter(ikwiad=ikwiad)
    loss_setter.set_hyperparameters(
        method=hyperparameters_config.loss.method,
        hyperparameters=hyperparameters_config.loss.hyperparams
    )
    criterion = loss_setter.get_loss().to(device=device)

    # set optimizer
    optim_setter = OptimSetter(ikwiad=ikwiad)
    optim_setter.set_hyperparameters(
        method=hyperparameters_config.optimizer.method,
        hyperparameters=hyperparameters_config.optimizer.hyperparams)
    optimizer = optim_setter.get_optim(parameters=model.parameters())

    # set saving
    save_name = name or training_config.save_params['save_name']
    os.makedirs(training_config.save_params['save_root'], exist_ok=True)
    model_path = os.path.join(
        f"{training_config.save_params['save_root']}", f"{save_name}"
    )
    try:
        # make save dir
        os.mkdir(model_path)
        if epoch and not ikwiad:
            warn("Could not find model to learn.")
    except FileExistsError:
        # file exists
        if epoch:
            ...
        confirm = (input(
            f"{save_name} already exists. Confirm overwrite with 'y': ").lower() == 'y'
        )
        if confirm:
            # reset save dir
            shutil.rmtree(model_path)
            sys.stdout.write(f"Removed existing model at {save_name}.\n")
            sys.stdout.flush()
            os.mkdir(model_path)
        else:
            # retain save dir
            sys.stdout.write(f"Exiting training and retaining {save_name}.\n")
            sys.stdout.flush()
            sys.exit(0)
    # save config
    shutil.copy(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'model_config.py'), model_path)
    os.sync()
    with open(os.path.join(model_path, 'model_config.py'), 'r') as f:
        lines = f.readlines()[5:]
    with open(os.path.join(model_path, 'model_config.py'), 'w') as f:
        top_level_doc = (
            f'r"""\nCNN build for {save_name}.\n\n'
            f'For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.\n"""\n'
        )
        f.write(top_level_doc)
        f.writelines(lines)
    os.sync()

    # training
    max_idx = len(loader)
    start_time = time.perf_counter()
    for ep in range(1, training_config.epochs + 1):
        if ep % training_config.save_gap == 0:
            # gap report
            sys.stdout.write(
                f"Epoch {str(ep).zfill(len(str(training_config.epochs)))} / {training_config.epochs}\n")
            sys.stdout.flush()
        # initial bar
        desc = (
            f"{str(0).zfill(len(str(max_idx)))}it/{max_idx}it  "
            f"{0:05.1f}%  "
            "?loss  "
            "?et  "
            "?eta  "
            "?it/s"
        )
        progress(-1, max_idx, b_len=75, desc=desc)
        running_loss = 0.0
        for i, data in enumerate(loader, start=1):
            # step
            inputs, labels = data
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # bar update
            running_loss += loss.item()
            elapsed = time.perf_counter() - start_time
            desc = (
                f"{str(i).zfill(len(str(max_idx)))}it/{max_idx}it  "
                f"{(100 * i / max_idx):05.1f}%  "
                f"{running_loss / i:.3}loss  "
                f"{convert_time(elapsed)}et  "
                f"{convert_time(elapsed * max_idx / i - elapsed)}eta  "
                f"{round(i / elapsed, 1)}it/s"
            )
            progress(i - 1, max_idx, b_len=75, desc=desc)

        if ep % training_config.save_gap == 0:
            # save gap model
            torch.save(model.state_dict(), os.path.join(model_path, f"epoch_{ep}"))
            os.sync()

    # save final model
    torch.save(model.state_dict(), os.path.join(model_path, f"epoch_{training_config.epochs}"))
    os.sync()

    return None
