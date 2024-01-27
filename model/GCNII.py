""" Pyg for “Simple and Deep Graph Convolutional Networks” """

import  torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch

class GCNII(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, alpha, theta, shared_weights, dropout):
        super(GCNII, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hid_channels))
        self.lins.append(nn.Linear(hid_channels, out_channels))
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers

        
        for layer in range(self.num_layers):
            self.convs.append(
                GCN2Conv(hid_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
     