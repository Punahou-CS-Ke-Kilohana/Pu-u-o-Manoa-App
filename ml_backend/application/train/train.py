r"""
This module consists of the training script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import time

from RHCCCore.network import CNNCore
from RHCCCore.utils import (
    OptimSetter,
    LossSetter,
    convert_time,
    progress
)

from ..configs.hyperparameters_config import hyperparameters_config
from ..configs.model_config import model_config
from ..dataloader.dataloader import loader
from ..configs.training_config import training_config


def train(ikwiad: bool = False) -> None:
    r"""
    Trains the model using the configs in the config folder.

    Args:
        ikwiad (bool, optional):
            "I know what I am doing" (ikwiad).
            If True, removes all warning messages.
            Defaults to False.

    Returns:
        None
    """
    # initialize model
    model = CNNCore(ikwiad=bool(ikwiad))
    model.set_channels(conv_channels=model_config.conv.sizes, dense_channels=model_config.dense.sizes)
    model.transfer_training_params(loader=loader)
    model.set_acts(methods=model_config.acts.methods, parameters=model_config.acts.params)
    model.set_conv(parameters=model_config.conv.conv_params)
    model.set_pool(parameters=model_config.conv.pool_params)
    model.set_dense(parameters=model_config.dense.params)
    model.instantiate_model(crossentropy=(hyperparameters_config.loss.method == 'CrossEntropyLoss'))

    # set loss
    loss_setter = LossSetter(ikwiad=ikwiad)
    loss_setter.set_hyperparameters(
        method=hyperparameters_config.loss.method,
        hyperparameters=hyperparameters_config.loss.hyperparams
    )
    criterion = loss_setter.get_loss()

    # set optimizer
    optim_setter = OptimSetter(ikwiad=ikwiad)
    optim_setter.set_hyperparameters(
        method=hyperparameters_config.optimizer.method,
        hyperparameters=hyperparameters_config.optimizer.hyperparams)
    optimizer = optim_setter.get_optim(parameters=model.parameters())

    # training
    max_idx = len(loader)
    start_time = time.perf_counter()
    for epoch in range(training_config.epochs):
        print(f"Epoch {epoch + 1} / {training_config.epochs}")
        desc = (
            f"{str(0).zfill(len(str(max_idx)))}it/{max_idx}it  "
            f"{0:05.1f}%  "
            "?loss  "
            "?et  "
            "?eta  "
            "?it/s"
        )
        progress(-1, max_idx, desc=desc)
        running_loss = 0.0
        for i, data in enumerate(loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            elapsed = time.perf_counter() - start_time
            desc = (
                f"{str(i + 1).zfill(len(str(max_idx)))}it/{max_idx}it  "
                f"{(100 * (i + 1) / max_idx):05.1f}%  "
                f"{running_loss / (i + 1):.3}loss  "
                f"{convert_time(elapsed)}et  "
                f"{convert_time(elapsed * max_idx / (i + 1) - elapsed)}eta  "
                f"{round((i + 1) / elapsed, 1)}it/s"
            )
            progress(i, max_idx, desc=desc)

    return None
