import os
import sys
import time

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from ..network.corecnn import CNNCore

from ..utils.visuals import (
    convert_time,
    progress
)

from ..configs.hyperparameters_config import hyperparameters_config
from ..configs.model_config import model_config
from ..configs.training_config import training_config


def train(ikwiad: bool = False):
    # initialize model
    model = CNNCore(ikwiad=ikwiad)
    model.set_channels(conv_channels=model_config.conv.sizes, dense_channels=model_config.dense.sizes)
    model.transfer_training_params(**model_config.training_params)
    model.set_acts(methods=model_config.acts.methods, parameters=model_config.acts.params)
    model.set_conv(parameters=model_config.conv.conv_params)
    model.set_pool(parameters=model_config.conv.pool_params)
    model.set_dense(parameters=model_config.dense.params)
    model.instantiate_model(crossentropy=(hyperparameters_config.loss.method == 'CrossEntropyLoss'))

    criterion = nn.CrossEntropyLoss(**hyperparameters_config.loss.params)
    optimizer = optim.Adam(model.parameters(), **hyperparameters_config.optimizer.params)

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
