import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from application import train_dataloader, test_dataloader  # Assume you have a separate test_dataloader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv block: 256 -> 252 -> 126
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # Second conv block: 126 -> 122 -> 61
        self.conv2 = nn.Conv2d(16, 32, 5)

        # Third conv block: 61 -> 57 -> 28
        self.conv3 = nn.Conv2d(32, 64, 5)

        # Fourth conv block: 28 -> 24 -> 12
        self.conv4 = nn.Conv2d(64, 64, 5)

        # Final feature map will be 64 x 12 x 12 = 9216
        self.fc1 = nn.Linear(64 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4
trainset = train_dataloader.dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = test_dataloader.dataset
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = [str(i) for i in list(range(15))]
iter_nums = []
logged_losses = []


def test_accuracy(net, testloader):
    # this is fried
    correct = 0
    total = 0
    net.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for testing
        for data in testloader:
            images, labels = data
            outputs = net(images)  # Shape should be (batch_size, 15)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Compare predictions with actual labels
    net.train()  # Set model back to training mode
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')




def main():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1 == 0:
                iter_nums.append(i + epoch * len(trainloader))
                logged_losses.append(running_loss)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
                running_loss = 0.0

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1} completed. Average loss: {running_loss / len(trainloader):.3f}')

    print('Finished Training')
    # test_accuracy(net, testloader)  # Evaluate test accuracy after training

    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pth')
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main()
    plt.plot(iter_nums, logged_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
