"""
GNN models for illicit transaction classification.

I went with GraphSAGE as the main model. Unlike GCN, SAGE learns an aggregation
function instead of relying on the full normalized adjacency matrix, so it's
inductive: a brand-new transaction can be scored from its neighborhood without
retraining. That's the property you need if transactions keep streaming in.

GCN is kept around for comparison.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be 2")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, 1)

    def embed(self, x, edge_index):
        """Node embeddings right before the classifier head (used to train the GAN)."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.relu(x)

    def forward(self, x, edge_index):
        # returns raw logits, one per node
        return self.classifier(self.embed(x, edge_index)).squeeze(-1)


class GCN(nn.Module):
    """Plain GCN, mostly here as a comparison point."""

    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be 2")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.convs[-1](x, edge_index))
        return self.classifier(x).squeeze(-1)
