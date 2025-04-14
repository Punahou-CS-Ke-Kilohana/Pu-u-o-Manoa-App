# Pu-u-o-Manoa-App ML Backend #

**This is the backend for the ML section of the app, coded in PyTorch**.
This file includes the core of the internal code, data to train on, saved models, and scripts to train and run the model.

----

## Table of Contents
- [Main Files](#main-files)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contact](#contact)

----

## Main Files
- **Application**: the internal settings and objects for specifically training the model.
- **Images**: The image processing and local images for training the model.
- **Models**: The trained models.
- **RHCCCore**: Generalized PyTorch code for building the model.
- **main.py**: Main execution code, used for training, validation.
- **method_selector.py**: Method selection for the execution.

----

## Usage

#### Scripts
The only file that should be run is *main.py*.
In the top line of the method_selector.py file, there will be a string specifying the method to be run in main.
Change this string to whatever you want the script to execute, and run main.
```python
method_name = 'train'
possible_methods = ['train', 'validate', 'interpret']
```
"train" trains a model.
"validate" validates a model by showing its statistics.
"interpret" interprets inputs with a trained model.
The files to edit settings for these files can be found under configs in the application folder.
To add files to train on, visit images and edit the file paths.

### Backend Basic Usage
The backend code was made to make creating models easier, but is still difficult to operate without knowing how PyTorch works.
Unless you know how to use the code, relying on the pre-made scripts applying it should be fine.
However, below is some basic application of the backend code.

#### Model and Algorithm Backend
This is how to instantiate a basic model using the backend code.
After this code, you can treat the model like a normal PyTorch model.
The loss and optim setter will give you the optimizer and loss objects from PyTorch to train the model.

```python
from RHCCCore.network import CNNCore
from RHCCCore.utils import OptimSetter, LossSetter

# initialize the model
model = CNNCore()
# sets the model channel sizes
model.set_channels(
    conv_channels=[16, 32, 64],
    dense_channels=[128, 64, 32]
)
# set the activation methods
model.set_acts()
# set the training parameters
model.transfer_training_params(
    color_channels=3,
    classes=15,
    initial_dims=(256, 256),
)
# set convolutional layers
model.set_conv()
# set the pooling layers
model.set_pool()
# set the dense layers
model.set_dense()
# instantiate model
model.instantiate_model()

# set loss
l_setter = LossSetter()
l_setter.set_hyperparameters()
criterion = l_setter.get_loss()

# set optimizer
o_setter = OptimSetter()
o_setter.set_hyperparameters()
optimizer = o_setter.get_optim(model.parameters())

```

#### Utils Backend
This is how to use the ParamChecker class to check and convert parameters.

```python
from RHCCCore.utils import ParamChecker

checker = ParamChecker(name='checker')

checker.set_types(
    default={'param1': 5},
    dtypes={'param1': (float, int)},
    vtypes={'param1': lambda x: 0 < x},
    ctypes={'param1': lambda x: float(x)}
)

test_params = {
    'param1': 3
}
checked_params = checker.check_params(test_params)

```

The progress bar and time converter are only used for visuals.
They are best applied in for loops.

```python
import time
from RHCCCore.utils import progress, convert_time

start_time = time.perf_counter()
bobs = ['bob'] * 5000

for i, bob in enumerate(bobs):
    elapsed_time = time.perf_counter() - start_time
    progress(
        idx=i,
        max_idx=len(bobs),
        desc=f"bob  et: {convert_time(elapsed_time)}"
    )
    time.sleep(0.01)

```

----

## Documentation
Documentation is located in comments and docstrings on most files.

----

## Contact
For questions, recommendations, or feedback, contact one of the app developers.

----
