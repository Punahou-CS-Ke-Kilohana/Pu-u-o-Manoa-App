import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt
import os

# Number of classes in your dataset
num_classes = 15

# Define the transformation for images, including resizing and converting to RGB
transform = transforms.Compose([
    transforms.Resize((256, 256)),              # Resize images to 256x256 to reduce memory usage
    transforms.ToTensor(),                        # Convert images to tensor
    transforms.ConvertImageDtype(torch.float),   # Ensure images are in float format
    transforms.Lambda(lambda img: img.permute(1, 2, 0))  # Convert from C x H x W to H x W x C
])

# Define the transformation for targets (one-hot encoding) using a named function
def one_hot_encode(y, num_classes=num_classes):
    one_hot = torch.zeros(num_classes, dtype=torch.float)
    return one_hot.scatter_(0, torch.tensor(y), value=1)


# Load your dataset using ImageFolder
training_data = datasets.ImageFolder(
    root=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pictures'),  # Set this to your training folder path
    transform=transform,
    target_transform=one_hot_encode  # Use the named function for target transformation
)
test_data = datasets.ImageFolder(
    root=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pictures'),  # Set this to your test folder path
    transform=transform,
    target_transform=one_hot_encode  # Use the named function for target transformation
)

# DataLoaders for batching the data, reduce batch size and use num_workers for efficiency
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

# Example to check batch shapes and visualize
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Feature space: ({train_features.size()[1]}, {train_features.size()[2]})")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Classes: {train_labels.size()[1]}")

# Visualize the first image in the batch
# img = train_features[0]  # Get the first image in the batch
# img = img.permute(0, 1, 2)  # Permute to shape (H, W, C)
# label = train_labels[0]
# plt.imshow(img)  # Now this will work without error
# plt.axis('off')  # Optionally turn off axis
# plt.show()

# Display label (from one-hot encoding)
# print(f"Label: {torch.argmax(label).item()}")  # Get the class index from one-hot encoded label