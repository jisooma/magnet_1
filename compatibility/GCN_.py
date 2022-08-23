# _*_codeing=utf-8_*_
# @Time:2022/7/20  17:03
# @Author:mazhixiu
# @File:GCN_.py
import sys
sys.path.append('/home/mzx/Code/MAAM')

import torch
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
from compatibility.conv.GCN_Conv import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.,aggregators=None):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        print(aggregators)
        self.convs.append(GCNConv(in_channels=in_channels, out_channels=hidden_channels,aggregators=aggregators))
        self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=out_channels,aggregators=aggregators))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, edge_index)

        return x.log_softmax(dim=-1)