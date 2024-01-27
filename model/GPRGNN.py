import torch.nn as nn
from torch_geometric.nn import MessagePassing
import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN

    ! 和Chebyshev的不同: 
    '''
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K              # * K = 10
        self.Init = Init        # * Init = PPR
        self.alpha = alpha      # * Texas  α = 1

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1) 
            TEMP[alpha] = 1.0    # [0, 1.0 ,0,0,0,0,0] 前K-1个γ都是0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)   # α(1-α)^0, α(1-α)^1, α(1-α)^2, α(1-α)^3,....   
            TEMP[-1] = (1-alpha)**K              
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)   # α^0, α^1, ... 
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random   10个 A_sum的系数随机初始化
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma  指定gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP, dtype=float))   # TEMP 初始化为APPNP的稀疏，作为初始化参数用于训练

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):  # 用PPR系数初始化 TEMP
            self.temp.data[k] = self.alpha*(1-self.alpha)**k  
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)  # edge_index,   A_sym

        hidden = x*(self.temp[0])  # γ_0 X
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)  # A_sym ^ (k+1) X
            gamma = self.temp[k+1]   # 每次diffusion的系数
            hidden = hidden + gamma*x  # 每次Diffusion乘各自系数  系数不同可以表示PPNP, 
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)



class GPRGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, K, beta, init, Gamma, dprate, dropout):
        super(GPRGNN,self).__init__()
        self.lin1 = nn.Linear(num_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)

        self.prop1 = GPR_prop(K, beta, init, Gamma)

        self.Init = init
        self.dprate = dprate  # prop dropout
        self.dropout = dropout     # linear prop dropout

    def forward(self, x, edge_index):
        # * Feature 先过MLP
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # * 
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x