import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from python_backend.core.dataloader import train_dataloader


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
        # Apply conv blocks with ReLU and pooling
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

classes = [str(i) for i in list(range(15))]

def main():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main()