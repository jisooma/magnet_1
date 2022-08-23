# _*_codeing=utf-8_*_
# @Time:2022/7/20  17:04
# @Author:mazhixiu
# @File:JK_NET_.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
# from torch_geometric.nn.conv import GCNConv
from compatibility.conv.GCN_Conv import GCNConv
from model import aggregators
from typing import List
from torch_geometric.typing import Tensor
from torch_geometric.nn import Linear

# 中间层用的GCN
class JK_NET(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.,jk='cat',aggregators=None,num_layers=4):
        super(JK_NET, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels=in_channels, out_channels=hidden_channels,aggregators=aggregators))

        for i in range(num_layers-1):
             self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=hidden_channels,aggregators=aggregators))

        # self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=out_channels, aggregators=aggregators))
        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if out_channels is not None:
            self.out_channels = out_channels
            if jk == 'cat':
                self.lin = Linear(num_layers * hidden_channels, out_channels)
            else:
                self.lin = Linear(hidden_channels, out_channels)
        else:
            if jk == 'cat':
                self.out_channels = num_layers * hidden_channels
            else:
                self.out_channels = hidden_channels

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x.log_softmax(dim=-1)
