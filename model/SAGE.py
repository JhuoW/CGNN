from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, n_cls, num_layers, dropout, norm, proj):
        super(SAGE, self).__init__()
        out_dim = hid_dim if proj else n_cls
        if num_layers == 1:
            self.convs = nn.ModuleList([SAGEConv(in_dim, out_dim)])
        else:
            self.convs = nn.ModuleList([SAGEConv(in_dim, hid_dim)])
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hid_dim, hid_dim))
            self.convs.append(SAGEConv(hid_dim, out_dim))
        if proj:
            self.lin = nn.Linear(hid_dim, n_cls)
        self.dropout = dropout
        self.proj = proj
        self.norm = norm
    
    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p = self.dropout, training = self.training)
                if self.norm:
                    h = F.normalize(h, p = 2, dim = 1)
        if self.proj:
            h = F.relu(h)
            h = F.dropout(h, p = self.dropout, training = self.training)
            h = self.lin(h)

        return h