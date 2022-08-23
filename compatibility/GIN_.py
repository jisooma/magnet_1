# _*_codeing=utf-8_*_
# @Time:2022/7/20  17:03
# @Author:mazhixiu
# @File:GIN_.py
import torch
import torch.nn.functional as F
# from torch_geometric.nn.conv import GINConv
from compatibility.conv.GIN_Conv import GINConv
from torch.nn import Sequential,Linear,BatchNorm1d,ReLU


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.,aggregators=None):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        # mlp1 = MLP([in_channels, out_channels, out_channels], batch_norm=True)
        # mlp2 = MLP([in_channels, out_channels, out_channels], batch_norm=True)
        self.convs.append(GINConv(
            GIN.MLP(in_channels=in_channels, out_channels=hidden_channels),aggregators=aggregators))
        self.convs.append(GINConv(GIN.MLP(in_channels=hidden_channels, out_channels=out_channels),
                                  aggregators=aggregators))

        # self.convs.append(GINConv(GIN.MLP(in_channels,hidden_channels)))
        # self.convs.append(GINConv(GIN.MLP(hidden_channels,out_channels)))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, edge_index)
        # print(x.shape)
        return x.log_softmax(dim=-1)
