import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# Load a dataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# Define a graph convolutional neural network
class Net(MessagePassing):
    def __init__(self):
        super(Net, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(dataset.num_features, 32)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        # print the row and col with the comment
        print("row ")
        print(row.shape)
        print("col") 
        print(col.shape)
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin(x)
        print("x now")
        print(x.shape)
        x = self.propagate(edge_index, x=x, norm=norm)
        print("x final")
        print(x.shape)
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

# Create a data loader
loader = DataLoader(dataset, batch_size=946, shuffle=True)

# Train the model
model = Net()
# explainer = torch_geometric.nn.models.Explainer(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for data in loader:
    optimizer.zero_grad()
    print(data.x.shape)
    print(data.y.shape)
    out = model(data.x, data.edge_index)
    print(out.size())
    loss = torch.nn.functional.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()