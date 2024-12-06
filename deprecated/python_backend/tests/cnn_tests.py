import sys
import os
import time

from application import CNNCore
from application import progress, convert_time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from application import train_dataloader, test_dataloader  # Assume you have a separate test_dataloader


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
    net = CNNCore()
    net.set_channels()
    net.transfer_training_params(3, 15, (256, 256))
    net.set_acts()
    net.set_conv()
    net.set_pool()
    net.set_dense()
    net.instantiate_model(crossentropy=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    start = time.time()

    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}")
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs)
            # print('----------------------------------')
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            elapsed = time.time() - start
            desc = (
                f"{str(i + 1).zfill(len(str(len(trainloader))))}it/{len(trainloader)}it  "
                f"{(100 * (i + 1) / len(trainloader)):05.1f}%  "
                f"{running_loss / (i + 1):.3}loss  "
                f"{convert_time(elapsed)}et  "
                f"{convert_time(elapsed * len(trainloader) / (i + 1) - elapsed)}eta  "
                f"{round((i + 1) / elapsed, 1)}it/s"
            )
            progress(i, len(trainloader), desc=desc)


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
