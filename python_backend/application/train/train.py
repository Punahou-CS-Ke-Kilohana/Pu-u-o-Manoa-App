import time

from core.network.corecnn import CNNCore

from core.utils.algorithms import (
    OptimSetter,
    LossSetter
)
from core.utils.utils import (
    convert_time,
    progress
)

from ..configs.hyperparameters_config import hyperparameters_config
from ..configs.model_config import model_config
from ..configs.training_config import training_config


def train(ikwiad: bool = False):
    r"""
    Trains the model using the

    Args:
        ikwiad:

    Returns:

    """
    # initialize model
    model = CNNCore(ikwiad=bool(ikwiad))
    model.set_channels(conv_channels=model_config.conv.sizes, dense_channels=model_config.dense.sizes)
    model.transfer_training_params(**model_config.training_params)
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
        hyperparameters=hyperparameters_config.optimizer.params)
    optimizer = optim_setter.get_optim(parameters=model.parameters())

    # training
    start_time = time.perf_counter()
    for epoch in range(training_config.training_params['epochs']):
        print(f"\nEpoch {epoch + 1}")
        running_loss = 0.0
        for i, data in enumerate(..., 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            elapsed = time.time() - start_time
            desc = (
                f"{str(i + 1).zfill(len(str(len(...))))}it/{len(...)}it  "
                f"{(100 * (i + 1) / len(...)):05.1f}%  "
                f"{running_loss / (i + 1):.3}loss  "
                f"{convert_time(elapsed)}et  "
                f"{convert_time(elapsed * len(...) / (i + 1) - elapsed)}eta  "
                f"{round((i + 1) / elapsed, 1)}it/s"
            )
            progress(i, len(...), desc=desc)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    train()
