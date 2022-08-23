# _*_codeing=utf-8_*_
# @Time:2022/7/9  10:10
# @Author:mazhixiu
# @File:utils.py
import logging.config
import sys

import json
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.data.dataset import get_train_val_test
from set_args import *
from scipy import sparse as sp

from torch_geometric.datasets import CitationFull, Coauthor,Amazon
import torch_geometric.utils as pygUtils


def judge_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        os.makedirs(path)

#19717
def cutDataSet(name):
    data = Dataset(root=clean_dataset_dir, name=name, seed=15, setting='gcn')
    print(data)
    num_node = len(data.labels)
    k = min(18000,num_node)
    keep_nodes = np.array([i for i in range(k)])
    features = data.features[0:k]
    labels = data.labels[0:k]
    edges = data.adj.nonzero()
    e0 = np.array(edges[0])
    e1 = np.array(edges[1])
    edge_index_array = np.array(list(zip(e0,e1)))
    print(edge_index_array)
    edge_index = torch.from_numpy(edge_index_array.T)

    # keep 18000 nodes
    nodes = torch.LongTensor(keep_nodes)
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
    print(data.adj.shape)
    print(type(data.adj))
    data.idx_train = idx_train
    data.idx_val = idx_val
    data.idx_test = idx_test

    # pubmed(adj_shape=(17999, 17999), feature_shape=(18000, 500))
    print(data)
    data_pyg = Dpr2Pyg(data)
    # Data(x=[18000, 500], edge_index=[2, 74393], y=[18000], train_mask=[18000], val_mask=[18000], test_mask=[18000])
    print(data_pyg.data)
    return data

def dir_(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

experiment_attention_dir = './attention_distribution'

def save_aggr_distribution(dataset,layer,aggregators,attention,aggr_max,attack=None,ptb=None):
    """
    保存某个数据集的聚合器的注意力分布
    :param dataset: 哪个数据集下的分布
    :param layer: 第几层的
    :param aggregators: 那几个聚合器的分布
    :param attention: 哪个注意力机制下的
    :param aggr_max: 分布是什么
    :param attack: 攻击
    :param ptb: 扰动比率
    :return:
    """
    if not os.path.exists(experiment_attention_dir):
        os.makedirs(experiment_attention_dir)
    if attack == None:
        np.savetxt("{}/{}_{}_{}_{}.txt".format(experiment_attention_dir, dataset, layer,aggregators,attention), np.array(aggr_max))
    else:
        np.savetxt("{}/{}_{}_{}_{}_{}.txt".format(experiment_attention_dir, dataset, layer,aggregators,attack,ptb), np.array(aggr_max))


experiment_output_dir = './output_representation'
def save_output(output,d_name,attention,aggregators,attack=None,ptb=None):

    if not os.path.exists(experiment_output_dir):
        os.makedirs(experiment_output_dir)
    if attack==None:
        np.save("{}/{}_{}_{}".format(experiment_output_dir, d_name, attention, aggregators),output)
    else:
        np.save("{}/{}_{}_{}_{}_{}".format(experiment_output_dir, d_name, attention, aggregators,attack,ptb), output)

# /home/mzx/Code/MAAM/clean_data
clean_dataset_dir ='/home/mzx/Code/MAAM_version_3/clean_data'
# target_attack_data_dir = '/home/mzx/Code/MAAM/attack_data/target_attack'
# global_attack_data_dir = '/home/mzx/Code/MAAM/attack_data/global_attack'
attack_data_dir ='/home/mzx/Code/MAAM_version_3/attack_data'

class Dataset_MAAM():
    """
    # 模仿dpr写的数据集读取形式
    """
    def __init__(self,clean_dir=clean_dataset_dir,dataset='cora',
                 attack=None,attack_dir=attack_data_dir,ptb=None):
        """

        :param clean_dir: 干净数据集保存目录
        :param dataset: 数据集
        :param attack: 攻击类型
        :param attack_dir: 攻击数据集保存目录
        :param ptb: 攻击扰动比率
        """
        self.clean_dir = clean_dir
        self.dataset = dataset.lower()
        self.attack = attack
        self.attack_dir = attack_dir
        self.ptb = ptb
        assert self.dataset in ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed',
                                'acm', 'blogcatalog', 'uai', 'flickr',
                                'cs','actor'], \
                'Currently only support cora, citeseer, cora_ml, polblogs, pubmed, acm, blogcatalog, flickr'

        if attack==None:
            self.adj, self.features, self.labels,self.idx_train,self.idx_val,\
                                        self.idx_test = self.load_clean_data(self.clean_dir,self.dataset)
        else:
            if ptb==None:
                assert 'please specify attck ptb'
            self.adj, self.features, self.labels,self.idx_train,self.idx_val,\
                                        self.idx_test = self.load_attack_data(self.dataset,self.attack,self.attack_dir,self.ptb)

    def load_clean_data(self,clean_dataset_dir,name):
        if name =='dblp':
            pyg_data = CitationFull(root=clean_dataset_dir, name=name)
        elif name == 'cs' or name=='physics':
            pyg_data = Coauthor(root=clean_dataset_dir,name=name)
        elif name =='computers' or name=='photo':
            pyg_data = Amazon(root=clean_dataset_dir,name=name)
        elif name =='cora' or name=='cora_ml'or name=='citeseer'or name=='polblogs'or name=='pubmed'\
                or name =='acm' or name=='flicker' or name=='uai' or name=='flickr':

            dpr_data = Dataset(root=clean_dataset_dir,name=name,seed=15)
            return dpr_data.adj,dpr_data.features,dpr_data.labels,dpr_data.idx_train,dpr_data.idx_val,dpr_data.idx_test
        else:
            assert name + "is not be supported！"

        labels = pyg_data.data.y.numpy()
        edge_index = pyg_data.data.edge_index
        features = pyg_data.data.x

        idx_train, idx_val, idx_test = get_train_val_test(
            len(labels), val_size=0.1, test_size=0.8, stratify=labels, seed=15)

        features = sp.csr_matrix(features.numpy())
        adj = pygUtils.to_scipy_sparse_matrix(edge_index).tocsr()
        return adj,features,labels,idx_train,idx_val,idx_test

    def load_attack_data(self,dataset, attack, attack_dir,ptb):

        # 参数seed=15
        # print('tt')
        data = Dataset(root=clean_dataset_dir, name=dataset, seed=15)
        features = data.features
        path = attack_dir + '/' + attack + '/' + dataset
        perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.npz'.format(path, attack, dataset, float(ptb)))
        data.adj = perturbed_adj
        adj = perturbed_adj
        idx_train = torch.LongTensor(data.idx_train)
        idx_val = torch.LongTensor(data.idx_val)
        idx_test = torch.LongTensor(data.idx_test)
        labels = torch.LongTensor(data.labels)

        row, col = np.diag_indices_from(adj)
        adj[row, col] = 1

        return adj,features, labels, idx_train, idx_val, idx_test

    def heterophily_handle(self,pyg_data,):
        n = pyg_data.num_nodes
        self.idx_train = self.mask_to_index(pyg_data.train_mask,n)
        self.idx_val = self.mask_to_index(pyg_data.val_mask, n)
        self.idx_test = self.mask_to_index(pyg_data.test_mask, n)

    def mask_to_index(self,index, size):
        all_idx = np.arange(size)
        return all_idx[index]

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2},labels={3},idx_train={4},idx_train={5},idx_train={6})'.format(
            'Dataset_MAAM:'+self.dataset, self.adj.shape, self.features.shape,
            self.labels.shape,self.idx_train.shape,self.idx_val.shape,self.idx_test.shape)



if __name__=='__main__':
    """
    各个工具函数的测试
    :return:
    """
    # for name in dataset_list:

    # load_Dataset('cora')
    for dataset in dataset_list:
        for attack in ['Nettack','RND']:
            acc_list = []
            for ptb in ['1.0', '2.0', '3.0', '4.0', '5.0']:
                data = Dataset_MAAM(dataset=dataset,attack=attack,ptb=ptb)
        # print(Dpr2Pyg(data).data)
        # print(len(Dpr2Pyg(data).data.y.unique()))