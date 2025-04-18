r"""
This module consists of the interpreting script for the Pu-u-o-Manoa-App.

For any questions or issues regarding this file, contact one of the Pu-u-o-Manoa-App developers.
"""

import csv
import importlib.util
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional

from RHCCCore.network import CNNCore

from application.configs.training_config import training_config

current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up from ml_backend/application/scripts to the root of Pu'u-o-Manoa-App
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))

# Construct the path to ImageCaptures
image_captures_path = os.path.join(project_root, "Pu'u-o-Manoa-App/Assets/ImageCaptures")


def interpret(
        device: torch.device = torch.device('cpu'),
        name: str = None,
        epoch: Optional[int] = None,
        ikwiad: bool = False
) -> None:
    r"""
    Interprets a trained model.

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
        CNNCore: Built model.
    """
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

    # import configs
    spec = importlib.util.spec_from_file_location('model_config', os.path.join(model_path, 'model_config.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_config = mod.model_config
    spec = importlib.util.spec_from_file_location('loader_config', os.path.join(model_path, 'loader_config.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    loader_config = mod.loader_config
    # make model
    model.set_channels(conv_channels=model_config.conv.sizes, dense_channels=model_config.dense.sizes)
    model.transfer_training_params(
        color_channels=loader_config.interpret_params['color_channels'],
        classes=loader_config.dataloader_params['classes'],
        initial_dims=loader_config.interpret_params['initial_dims']
    )
    model.set_acts(methods=model_config.acts.methods, parameters=model_config.acts.params)
    model.set_conv(parameters=model_config.conv.conv_params)
    model.set_pool(parameters=model_config.conv.pool_params)
    model.set_dense(parameters=model_config.dense.params)
    model.instantiate_model(crossentropy=False)
    # load model
    model.load_state_dict(torch.load(os.path.join(model_path, f'epoch_{epoch}'), weights_only=True))
    model = model.to(device=device)
    model.eval()

    # transformation
    transform = transforms.Compose([
        transforms.Resize(loader_config.interpret_params['initial_dims']),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    # label names
    with open(os.path.join(model_path, 'label_names.csv'), 'r', newline='') as f:
        reader = csv.reader(f)
        labels = next(reader)

    while True:
        # todo: this is fried as hell
        # 

        current_dir = os.path.dirname(os.path.abspath(__file__))

        image_dir = os.path.join(current_dir, "../../../Pu'u-o-Manoa-App/Assets/ImageCaptures")
        image_dir = os.path.abspath(image_dir)

        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(image_dir, f))]

        if not image_files:
            raise FileNotFoundError(f"No image files found in directory: {image_dir}")

        # Get the most recently jpg
        latest_image = max(image_files, key=os.path.getmtime)

        img = Image.open(latest_image)
        pred = model.forward_no_grad(x=transform(img).to(device=device))
        print(labels[int(torch.argmax(pred))])
        return labels[int(torch.argmax(pred))]


if __name__ == "__main__":
    intp_device = torch.device('cpu')
    intp_name = 'test_model'
    interpret(device=intp_device, name=intp_name)
