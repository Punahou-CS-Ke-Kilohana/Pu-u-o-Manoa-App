# do NOT touch this unless you REALLY know what you're doing

from easydict import EasyDict as Edict

hyperparameters_config = Edict()

hyperparameters_config.loss = Edict()
hyperparameters_config.loss.method = 'CrossEntropyLoss'
hyperparameters_config.loss.params = {
    'weight': None,
    'size_average': None,
    'ignore_index': -100,
    'reduce': None,
    'reduction': 'mean',
    'label_smoothing': 0.0
}

hyperparameters_config.optimizer = Edict()
hyperparameters_config.optimizer.method = 'Adam'
hyperparameters_config.optimizer.params = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False,
    'foreach': None,
    'maximize': False,
    'capturable': False,
    'differentiable': False,
    'fused': None
}
