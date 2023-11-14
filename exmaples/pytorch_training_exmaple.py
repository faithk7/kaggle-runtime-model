import torch
import torch.nn as nn
import torch.optim as optim


# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Prepare your data
inputs = torch.randn(256, 10)
labels = torch.randint(0, 2, (256,))

# Train the model
for epoch in range(10):  # loop over the dataset multiple times

    # Forward pass: compute predicted y by passing x to the model
    outputs = net(inputs)

    # Compute the loss
    loss = criterion(outputs, labels)

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch: {} Loss: {:.3f}'.format(epoch+1, loss.item()))