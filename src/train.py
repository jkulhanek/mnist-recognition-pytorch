import torch
import torch.optim as optim
import torch.nn as nn
from model import Net
import dataset

learningRate = 0.01
epochs = 2

net = Net()
optimizer = optim.SGD(net.parameters(), lr=learningRate)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataset.trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Store weights
torch.save(net.state_dict(), 'net.pth')

# Run the test
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset.testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

test()