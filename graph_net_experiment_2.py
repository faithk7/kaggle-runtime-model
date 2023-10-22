import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset  
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool 

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=32, shuffle=True) 
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)


    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  

    model.train()
    for epoch in range(200):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()        
            optimizer.step()

    for data in test_loader:
        model.eval()
        _, pred = model(data.x, data.edge_index).max(dim=1)
        correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))
