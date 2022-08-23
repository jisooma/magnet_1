# _*_codeing=utf-8_*_
# @Time:2022/8/2  09:51
# @Author:mazhixiu
# @File:test_combination_attack.py
import sys

sys.path.append('/home/mzx/Code/MAAM_version_2')

import torch
import torch.nn.functional as F

from utils.utils import Dataset_MAAM
from deeprobust.graph.data import Dpr2Pyg
from GCN_ import GCN
from GAT_ import GAT
from SAGE_ import SAGE
from JK_NET_ import JK_NET
from GIN_ import GIN
import numpy as np
import os
from model.aggregators import AGGREGATORS

def train(data):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[data.train_mask]  # 前面我们提到了，GCN是实现了edge_index和adj_t两种形式的
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(data):
    model.eval()

    out = model(data.x, data.edge_index)
    y_pred = out.argmax(axis=-1)

    correct = y_pred == data.y
    train_acc = correct[data.train_mask].sum().float() / data.train_mask.sum()
    valid_acc = correct[data.val_mask].sum().float() / data.val_mask.sum()
    test_acc = correct[data.test_mask].sum().float() / data.test_mask.sum()

    return train_acc, valid_acc, test_acc


path_acc = './single_acc_attack_1/'

def judge_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        os.makedirs(path)

# , 'SGAttack', 'RND', 'FGA'

# """
# 1、单聚合器：
# sum
# mean
# max
# trimmed

# min
# mode
# median
# """
#
Aggregators = [['sum'],['mean'],['min'],['max'],['mode'],['median'],['trimmed']]
"""
2、双聚合器性能排序
"""
# Aggregators = [['sum','mean'],['sum','max'],['sum','trimmed'],
#                ['mean','max'],['mean','max'],['max','trimmed']]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 5*5*8*5*7
# 'GCN','GAT','JK_NET', 'GIN''RND',
# '1.0', '2.0', '3.0',
#  'SAGE',
# 'cora','cora_ml', 'citeseer',
# 'cora_ml', 'citeseer', 'polblogs', 'pubmed'
# '1.0', '2.0','3.0',
# '4.0', '5.0'
#'Nettack',
# 'citeseer', 'polblogs',
# 'cora_ml', 'citeseer', 'polblogs', 'pubmed'
# 'Nettack',
# 'GCN','GAT','cora',
# 'cora','cora_ml',
# 'JK_NET','Metattack'
# 'cora_ml','citeseer',
# 'citeseer','polblogs','pubmed','cora',
# '1.0', '2.0','3.0', '4.0',
# 'pubmed',
# 'SAGE','GIN','GCN','GAT',
#'SAGE',# 'cora_ml','citeseer','polblogs','pubmed'
# '0.10','0.15','0.20','0.25'
# 'PGDAttack'
for fun in ['JK_NET','GIN','SAGE']:
    for dataset in ['cora','cora_ml','citeseer','polblogs','pubmed',]:
        for attack in ['Metattack']:
        # for attack in ['']:
            for ptb in ['0.05','0.10','0.15','0.20','0.25']:
        #     for ptb in ['5.0']:
                acc_list = []
                for aggregator in Aggregators:
                    data = Dataset_MAAM(dataset=dataset,attack=attack,ptb=ptb)
                    pyg_data = Dpr2Pyg(data).data.to(device)
                    model = eval(fun)(in_channels=data.features.shape[1], hidden_channels=128,
                                      out_channels=data.labels.max().item() + 1, aggregators=aggregator).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

                    max_acc = 0
                    for epoch in range(500):
                        loss = train(pyg_data)
                        train_acc, valid_acc, test_acc = test(pyg_data)

                        if test_acc > max_acc:
                            max_acc = test_acc
                        print(f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train_acc: {100 * train_acc:.3f}%, '
                              f'Valid_acc: {100 * valid_acc:.3f}% '
                              f'Test_acc: {100 * test_acc:.3f}%')
                    acc_list.append(max_acc.cpu().numpy())
                path = path_acc+'/global'
                judge_dir(path)
                np.savetxt("{}/{}_{}_{}_{}.txt".format(path, fun, dataset,attack,ptb), np.array(acc_list))