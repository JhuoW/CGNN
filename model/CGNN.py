from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
)
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import sum as sparsesum
from torch_sparse import mul
import torch.nn.functional as F

def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A} 
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))

def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj



def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)        
    else:
        raise ValueError(f"{norm} normalization is not supported")

class MyGCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, conv_norm) -> None:
        super(MyGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(input_dim, output_dim)
        self.conv_norm = conv_norm
        self.adj_norm = None
        self.C = None
    
    def forward(self, x, edge_index, C = None):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, self.conv_norm)

        
        # prop_x = self.adj_norm @ x
        prop_x  = C @ x
        trans_x = self.lin(prop_x)
        return trans_x


class CGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, n_cls, num_layers, dropout, jk, norm, conv_norm):
        super(CGNN, self).__init__()
        out_dim = hid_dim if jk else n_cls

        if num_layers == 1:
            self.convs = nn.ModuleList([MyGCNConv(in_dim, out_dim, conv_norm)])
        else:
            self.convs = nn.ModuleList([MyGCNConv(in_dim, hid_dim, conv_norm)])
            for _ in range(num_layers - 2):
                self.convs.append(MyGCNConv(hid_dim, hid_dim, conv_norm))
            self.convs.append(MyGCNConv(hid_dim, out_dim, conv_norm))
        
        if jk is not None:
            jk_in_dim = hid_dim * num_layers if jk == "cat" else hid_dim
            self.jk_lin = nn.Linear(jk_in_dim, n_cls)
            self.jump = JumpingKnowledge(mode=jk, channels=hid_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk
        self.norm = norm
    
    def forward(self, x, edge_index, C):
        h = x
        hs = []
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, C)
            if i != len(self.convs) - 1 or self.jk:
                h = F.relu(h)
                h = F.dropout(h, p = self.dropout, training = self.training)
                if self.norm:
                    h = F.normalize(h, p = 2, dim = 1)
            hs += [h]
        if self.jk is not None:
            h = self.jump(hs)
            h = self.jk_lin(h)

        return h