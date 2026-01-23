import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F
from utils import create_activation  # Import create_activation from graphmae.utils

def dgl_to_torch_geom(dgl_graph):
    # Extract edge indices and stack them
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index

class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes, K=2, dropout=0.5, activation=None, residual=False, norm=None, cached=False):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=K, cached=cached)
        self.dropout = dropout
        self.activation = create_activation(activation)  # Use create_activation from graphmae.utils
        self.residual = residual
        self.head = torch.nn.Identity()

        if norm is not None:
            self.norm = norm(num_classes)
        else:
            self.norm = torch.nn.Identity()

        if residual:
            self.res_fc = torch.nn.Linear(num_features, num_classes, bias=False)
        else:
            self.res_fc = torch.nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        edge_index = dgl_to_torch_geom(g)

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)

        if self.residual:
            h = h + self.res_fc(inputs)

        h = self.norm(h)

        # if self.activation is not None:
        #     h = self.activation(h)

        hidden_list.append(h)

        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)