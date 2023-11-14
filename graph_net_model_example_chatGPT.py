import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load the CORA dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Perform 1st convolution
        x = self.conv1(x, edge_index)
        # Apply ReLU activation
        x = F.relu(x)
        # Apply dropout
        x = F.dropout(x, training=self.training)
        # Perform 2nd convolution
        x = self.conv2(x, edge_index)

        # Return the log softmax output
        return F.log_softmax(x, dim=1)

# Initialize the model, optimizer, and data
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
data = dataset[0]

# Define the training function
def train():
    # Set the model to train mode
    model.train()
    # Zero the gradients
    optimizer.zero_grad()
    # Perform a forward pass
    out = model(data)
    # Calculate the loss
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # Perform a backward pass
    loss.backward()
    # Update the parameters
    optimizer.step()

# Define the testing function
def test():
    # Set the model to evaluation mode
    model.eval()
    # Perform a forward pass
    out = model(data)
    # Get the predictions
    pred = out.argmax(dim=1)
    # Calculate the number of correct predictions
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    # Calculate the accuracy
    acc = correct / data.test_mask.sum().item()
    # Return the accuracy
    return acc

# Train for 200 epochs
for epoch in range(2000):
    # Perform a training pass
    train()
    # Print the accuracy every 10 epochs
    if (epoch+1) % 10 == 0:
        test_acc = test()
        print(f'Epoch: {epoch+1}, Test Accuracy: {test_acc}')