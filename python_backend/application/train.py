from python_backend.core.cnn import TorchCNN
from python_backend.core.dataloader import train_dataloader

default_pool = {
    'kernel_size': 3,
    'stride': None,
    'padding': 0,
    'dilation': 1,
    'return_indices': False,
    'ceil_mode': False
}

default_conv = {
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
    'bias': True,
    'padding_mode': 'zeros'
}
default_dense = {
    'bias': True
}

default_relu = {
    'inplace': False
}

default_softmax = {
    'dim': None
}

default_centropy = {
    'weight': None,
    'size_average': True,
    'ignore_index': -100,
    'reduce': True,
    'reduction': 'mean',
    'label_smoothing': 0.0
}

default_adam = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0.0,
    'amsgrad': False,
    'foreach': None,
    'maximize': False,
    'fused': False
}

rhcc_cnn = TorchCNN(status_bars=True)

rhcc_cnn.set_sizes()  # 4 conv, 6 dense

rhcc_cnn.configure_network(train_dataloader)

rhcc_cnn.set_conv(parameters=4 * [default_conv])
rhcc_cnn.set_pool(parameters=4 * [default_pool])
rhcc_cnn.set_dense(parameters=6 * [default_dense])
rhcc_cnn.set_loss(parameters=default_centropy)
rhcc_cnn.set_optim(parameters=default_adam)

rhcc_cnn.fit(
    parameters={
        'epochs': 5
    }
)
