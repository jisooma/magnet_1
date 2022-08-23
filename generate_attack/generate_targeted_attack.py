# _*_codeing=utf-8_*_
# @Time:2022/4/1  18:16
# @Author:mazhixiu
# @File:generate_targeted_attack.py

# import torch.cuda
import sys
sys.path.append('/home/mzx/Code/MAAM_version_2')
from deeprobust.graph.defense import GCN,SGC
from deeprobust.graph.targeted_attack import Nettack,FGA,IGAttack,SGAttack,RND
import random
from utils.utils import *
import scipy.sparse as sp
from torch_geometric.datasets import CitationFull
from deeprobust.graph.data import Dataset, Dpr2Pyg
from set_args import *
import json

import os

path = os.getcwd()
# 19717
from deeprobust.graph.data.dataset import get_train_val_test
import torch_geometric.utils as pygUtils

class DBLP_dpr():
    def __init__(self):
        self.adj, self.features, self.labels,self.idx_train,self.idx_val,self.idx_test = self.load_data()

    def load_data(self):

        pyg_data = CitationFull(root=clean_dataset_dir, name='dblp')
        print(pyg_data.data)
        # Data(x=[17716, 1639], edge_index=[2, 105734], y=[17716])
        labels = pyg_data.data.y.numpy()
        edge_index = pyg_data.data.edge_index
        features = pyg_data.data.x

        idx_train, idx_val, idx_test = get_train_val_test(
            len(labels), val_size=0.1, test_size=0.8, stratify=labels, seed=15)

        features = sp.csr_matrix(features.numpy())
        # features = pygUtils.to_scipy_sparse_matrix(features).tocsr()
        adj = pygUtils.to_scipy_sparse_matrix(edge_index).tocsr()
        print(features.shape)
        print(type(features))
        print(adj.shape)
        print(type(adj))
        print(type(idx_train))
        return adj,features,labels,idx_train,idx_val,idx_test

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format('DBLP', self.adj.shape, self.features.shape)

def cutDataSet(name):
    data = Dataset(root=clean_dataset_dir, name=name, seed=15, setting='gcn')
    print(data)
    num_node = len(data.labels)
    k = min(18100,num_node)
    # print(k)
    keep_nodes = np.array([i for i in range(k)])
    # print(len(keep_nodes))
    features = data.features[0:k]
    # print(len(features))
    labels = data.labels[0:k]
    # print(len(labels))
    # print(data.adj)
    edges = data.adj.nonzero()

    e0 = np.array(edges[0])
    e1 = np.array(edges[1])
    # print(len(e1))
    edge_index_array = np.array(list(zip(e0,e1)))
    # print(edge_index_array)
    edge_index = torch.from_numpy(edge_index_array.T)

    # keep 18000 nodes
    nodes = torch.LongTensor(keep_nodes)
    print(nodes)
    edge_index = pygUtils.subgraph(nodes,torch.LongTensor(edge_index.long()))[0]
    print(edge_index)
    # adj = edge_index
    idx_train, idx_val, idx_test =get_train_val_test(
        len(labels), val_size=0.1, test_size=0.8, stratify=labels, seed=15)

    data.features = features
    data.labels = labels
    # num_edge  = len(edge_index[0])
    # print(np.ones(num_edge).shape)
    # # print(np.diag())
    # data.adj = sp.csr_matrix(np.ones(num_edge),(edge_index[0].int(),edge_index[1].int())).shape(num_node,num_node)
    data.adj = pygUtils.to_scipy_sparse_matrix(edge_index).tocsr()
    # print(data.adj.shape)
    # print(type(data.adj))
    data.idx_train = idx_train
    data.idx_val = idx_val
    data.idx_test = idx_test

    # pubmed(adj_shape=(17999, 17999), feature_shape=(18000, 500))
    print(data)
    # data_pyg = Dpr2Pyg(data)
    return data


def select_attack_nodes(save_dir=None,dataset=None,dataset_name=None):
    adj = dataset.adj
    idx_test=dataset.idx_test

    degrees = adj.A.sum(0)

    adj1 = adj
    po_edges = []

    # 选择度大于10的节点进行攻击
    for i in range(idx_test.size):
        if degrees[i] > 10:
            po_edges.append(i)

    po_edges = random.sample(po_edges, int(len(po_edges) * 0.5))
    print(os.path.join(save_dir,dataset_name))
    with open('{}/{}_attacked_nodes_1.json'.format(save_dir,dataset_name), 'w') as fp:
        json.dump({"attacked_test_nodes": po_edges}, fp)
    return po_edges


def set_surrogate_model(data,attack):

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    if attack=='SGAttack':
        # Setup Surrogate conv
        surrogate = SGC(nfeat=features.shape[1],
                        nclass=labels.max().item() + 1, K=2,
                        lr=0.01, device=device).to(device)

        pyg_data = Dpr2Pyg(data).pyg_data
        surrogate.fit(pyg_data, verbose=False)  # train with earlystopping
        surrogate.test()

    else:
        # set surrogate conv
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate

def set_attack_model(attack,target_node,surrogate,data,j):
    # print(attack)
    # print(data)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    if attack == 'Nettack':
        """
        可以攻击图结构和节点特征
        """
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device).to(
            device)
        # Attack
        model.attack(features, adj, labels, target_node, n_perturbations=j+1)
    elif attack == 'FGA':
        """
        只能攻击图结构
        """
        model = FGA(surrogate, nnodes=adj.shape[0], device=device).to(device)
        # Attack
        model.attack(features, adj, labels, idx_train, target_node, n_perturbations=j + 1)
    elif attack =='IGAttack':
        """
         可以攻击图结构和节点特征
        """
        model = IGAttack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, target_node, j+1, steps=20)
    elif attack =='SGAttack':
        """
        可以攻击图结构和节点特征
        """
        model = SGAttack(surrogate, attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, j+1, direct=True)
    elif attack =='RND':
        model = RND()
        model.attack(adj,labels,idx_train,target_node,n_perturbations=j+1)
    else:
        model = surrogate

    modified_adj = model.modified_adj
    modified_features = model.modified_features
    return modified_adj,modified_features

def generate_targeted_attack(save_dir=adv_dataset_dir):

    if not os.path.exists(adv_dataset_dir):
        os.makedirs(adv_dataset_dir)
    # 'IGAttack'# 'Nettack','SGAttack','RND','Nettack','FGA',
    #
    targeted_attack_list = ['SGAttack']
    for attack in targeted_attack_list:
        print("--------attack-----:",attack)
        # 'cora','cora_ml','citeseer','polblogs',
        for name in ['pubmed']:
            print("--------dataset-----:", name)
            """
            data:干净的数据
            po_edges:攻击的节点
            data_adv:被攻击的数据
            """
            if attack=='IGAttack' and name=='Pubmed':
                data = cutDataSet(name)
            else:
                data = Dataset_MAAM(dataset =name)

            attack_data_save_dir = adv_dataset_dir+'/'+attack
            if not os.path.exists(attack_data_save_dir):
                os.makedirs(attack_data_save_dir)
            po_edges = select_attack_nodes(attack_data_save_dir,data,name)
            data_adv = data
            for j in range(5):
                print("--------pertubation-----:", j+1)
                for i in range(len(po_edges)):
                    # Setup Surrogate conv
                    # if attack!='RND':
                    surrogate = set_surrogate_model(data_adv,attack)
                    target_node = po_edges[i]
                    # Setup Attack Model
                    modified_adj,modified_features= set_attack_model(attack,target_node,surrogate,data_adv,j)
                    adj = sp.csr_matrix(modified_adj.A)
                    data_adv.adj = adj

                if not os.path.exists(os.path.join(attack_data_save_dir,name)):
                    os.makedirs(os.path.join(attack_data_save_dir,name))
                print(os.path.join(attack_data_save_dir,name))
                sp.save_npz('{}/{}/{}_{}_adj_{}.0'.format(attack_data_save_dir,name,attack,name, j+1), data_adv.adj)


generate_targeted_attack(adv_dataset_dir)

