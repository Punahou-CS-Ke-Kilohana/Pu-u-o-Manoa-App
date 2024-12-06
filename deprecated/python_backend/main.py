import torch.multiprocessing as mp

from .application.train.train import train


# multiprocessing
mp.freeze_support()
mp.set_start_method('spawn', force=True)
# train
train()
