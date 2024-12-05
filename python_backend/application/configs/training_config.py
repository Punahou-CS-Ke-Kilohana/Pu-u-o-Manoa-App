import os
import datetime
from easydict import EasyDict as Edict

training_config = Edict()

training_config.save_params = {
    'save_root': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'),
    'save_name': f"model_{datetime.datetime.now().year}_{datetime.datetime.now().month}_{datetime.datetime.now().day}"
}

training_config.dataloader_params = {
    'batch_size': 5
}

training_config.training_params = {
    'epochs': 5
}
