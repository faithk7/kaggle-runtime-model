import torch
import torch.nn as nn
import torch.nn.functional as F


class _OpEmbedding(nn.Module):
    """Embeds operation nodes into feature 'op_e'."""

    def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_ops, embed_d)
        # In PyTorch, L2 regularization is typically applied during optimization, not in the layer

    def forward(self, graph):
        # Assuming graph is a PyTorch-Geometric-style data object
        op_features = graph.x  # Assuming x contains the features of operation nodes
        op_indices = (
            graph.op_indices
        )  # Assuming op_indices contains the indices of operation nodes
        op_features_embedded = self.embedding_layer(op_indices)
        graph.x = op_features_embedded
        return graph


def _mlp(layer_dims, hidden_activation):
    """Creates an MLP with given layer dimensions and activation."""
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            if hidden_activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            # Add more activation functions if needed
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Embeds op codes, averages features across all-nodes, passing thru MLP."""

    def __init__(
        self,
        num_configs: int,
        num_ops: int,
        op_embed_dim: int = 32,
        mlp_layers: int = 2,
        hidden_activation: str = "leaky_relu",
        hidden_dim: int = 64,
        reduction: str = "sum",
    ):
        super().__init__()
        self.num_configs = num_configs
        self.num_ops = num_ops
        self.op_embedding = _OpEmbedding(num_ops, op_embed_dim)
        self.reduction = reduction
        layer_dims = [hidden_dim] * mlp_layers
        layer_dims.append(1)
        self.mlp = _mlp(layer_dims, hidden_activation)

    def forward(self, graph):
        # Assuming graph is a PyTorch-Geometric-style data object
        graph = self.op_embedding(graph)

        # Implement node pooling for the 'op' node set
        if self.reduction == "sum":
            op_feats = torch.sum(graph.x, dim=0)
        # Add other reduction types if needed

        # Implement feature processing for 'config' node set
        # This part of the code will heavily depend on the structure of your graph data
        config_feats = (
            graph.config_feats
        )  # Assuming config_feats is a tensor of config node features
        batch_size = config_feats.size(0)
        config_feats = config_feats.view(batch_size, -1, config_feats.size(-1))

        # Combine features
        op_feats = op_feats.unsqueeze(0).expand(batch_size, self.num_configs, -1)
        combined_feats = torch.cat([op_feats, config_feats], dim=-1)

        return self.mlp(combined_feats).squeeze(-1)


# The rest of the code will remain the same.
