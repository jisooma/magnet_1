# _*_codeing=utf-8_*_
# @Time:2022/7/19  18:53
# @Author:mazhixiu
# @File:GIN_Conv.py
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from torch_geometric.nn.inits import reset

from model.aggregators import AGGREGATORS

class GINConv(MessagePassing):

    def __init__(self, nn: Callable,
                 eps: float = 0.,
                 train_eps: bool = False,
                 aggregators = None,
                 attentions:bool=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.aggregators = aggregators
        self.with_attention = attentions

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    # 多聚合器
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        if self.aggregators==None:
            return AGGREGATORS['sum'](inputs, index, dim_size)

        outs = [AGGREGATORS[aggr](inputs,index,dim_size) for aggr in self.aggregators]

        out_stack = torch.stack(outs, dim=1)  # N*M*D' M是聚合器数量

        if self.with_attention:
            result, aggr_max = self.attention(out_stack, out_stack, out_stack)  # 在注意层，所有聚合器的聚合结果经过注意力机制之后变成了 N*F‘
            self.aggr_max = aggr_max
            # save_aggr_distribution(self.dataset, self.layer, self.aggr_max.cpu().numpy().astype(int))
        else:
            result = out_stack.sum(1)

        return result

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
