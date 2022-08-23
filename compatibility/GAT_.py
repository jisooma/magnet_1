# _*_codeing=utf-8_*_
# @Time:2022/7/20  17:03
# @Author:mazhixiu
# @File:GAT_.py
import torch
import torch.nn.functional as F
# from torch_geometric.nn.conv import GATConv
from compatibility.conv.GAT_Conv import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.,aggregators=None):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels=in_channels, out_channels=hidden_channels,aggregators=aggregators))
        self.convs.append(GATConv(in_channels=hidden_channels, out_channels=out_channels,aggregators=aggregators))
        # self.convs.append(GATConv(in_channels=in_channels, out_channels=hidden_channels))
        # self.convs.append(GATConv(in_channels=hidden_channels, out_channels=out_channels))

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
