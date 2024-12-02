from easydict import EasyDict as Edict

training_config = Edict()

training_config.dataloader_params = {
    'batch_size': 5
}

training_config.training_params = {
    'epochs': 5
}
