# _*_codeing=utf-8_*_
# @Time:2022/7/19  19:16
# @Author:mazhixiu
# @File:SAGE_Conv.py

from typing import Union,Tuple
from torch_geometric.typing import OptPairTensor,Adj,Size

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch
from typing import Optional

from model.aggregators import AGGREGATORS

class SAGEConv(MessagePassing):
    def __init__(self,in_channels:Union[int,Tuple[int,int]],
                 out_channels:int,normalize:bool=False,
                 root_weight:bool=True,
                 aggregators = None,
                 with_attention = False,
                 bias:bool=True,**kwargs):
        kwargs.setdefault('aggr','mean')

        super(SAGEConv,self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.aggregators = aggregators
        self.with_attention = with_attention
        self.root_weight = root_weight

        if isinstance(in_channels,int):
            in_channels = (in_channels,in_channels)

        self.lin_l = Linear(in_channels[0],out_channels,bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1],out_channels,bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self,x:Union[Tensor,OptPairTensor],edge_index:Adj,size:Size=None)->Tensor:

        if isinstance(x,Tensor):
            x:OptPairTensor = (x,x)

        out = self.propagate(edge_index,x=x,size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out,p=2,dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    # 多聚合器
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:

        if self.aggregators == None:
            return AGGREGATORS[self.aggr](inputs, index, dim_size)

        outs = [AGGREGATORS[aggr](inputs,index,dim_size) for aggr in self.aggregators]

        out_stack = torch.stack(outs, dim=1)  # N*M*D' M是聚合器数量

        if self.with_attention:
            result,aggr_max = self.attention(out_stack,out_stack,out_stack) # 在注意层，所有聚合器的聚合结果经过注意力机制之后变成了 N*F‘
            self.aggr_max = aggr_max
            # save_aggr_distribution(self.dataset, self.layer, self.aggr_max.cpu().numpy().astype(int))
        else:
            result = out_stack.sum(1)

        return result

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        if self.aggregators == None:
            return matmul(adj_t, x,reduce=self.aggr)

        out_list = []
        for aggr in self.aggregators:
            out = matmul(adj_t, x,reduce=aggr)
            out_list.append(out)
        out_stack = torch.stack(out_list, dim=1)  # N*M*D' M是聚合器数量

        if self.with_attention:
            result, aggr_max = self.attention(out_stack, out_stack, out_stack)  # 在注意层，所有聚合器的聚合结果经过注意力机制之后变成了 N*F‘
            self.aggr_max = aggr_max
            # save_aggr_distribution(self.dataset, self.layer, self.aggr_max.cpu().numpy().astype(int))
        else:
            result = out_stack.sum(1)
        return result

