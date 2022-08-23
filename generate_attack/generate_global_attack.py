# _*_codeing=utf-8_*_
# @Time:2022/4/1  21:46
# @Author:mazhixiu
# @File:generate_global_attack.py


import sys
sys.path.append('/home/mzx/Code/MAAM_version_3/')

from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack,MinMax,DICE,Random,Metattack
from deeprobust.graph.utils import preprocess
from set_args import *
import scipy.sparse as sp
import os
from utils.utils import Dataset_MAAM


def set_surrogate_model(data):
    # set surrogate conv
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate

"""
Randomly adding edges to the input graph
"""
def generate_random_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir,'Random',dataset)

    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)
    for i in range(5):
        data=Dataset_MAAM(dataset=dataset)
        adj, features, labels = data.adj, data.features, data.labels
        model = Random(attack_features=False,attack_structure=True)

        ptb_rate = 0.05 * (i + 1)
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(adj,n_perturbations=n_perturbations)
        modified_adj = model.modified_adj

        save_random_dir ='{}/Random_{}_adj_{}'.format(adj_dir,dataset,round(ptb_rate,3))
        # save_random_dir = '{}/Random_{}_features_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        sp.save_npz(save_random_dir,sp.csr_matrix(modified_adj))


"""
dice:
"""
def generate_dice_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir,'Dice',dataset)
    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)
    for i in range(5):
        data = Dataset_MAAM(dataset=dataset)
        adj,features,labels = data.adj,data.features,data.labels
        model = DICE()
        """
        n_perturbations：边增加或者移动的数量
                0.05*[1,2,3,4,5]
        扰动比率：[0.05,0.1,0.15,0.2,0.25]
        扰动数量：int(0.05 * (i + 1) * (adj.sum() // 2))
        """
        ptb_rate = 0.05 * (i + 1)
        n_perturbations = int(ptb_rate* (adj.sum() // 2))
        model.attack(adj,labels,n_perturbations=n_perturbations)
        modified_adj = model.modified_adj
        # 对抗样本数据集保存
        save_dice_dir = '{}/Dice_{}_adj_{}'.format(adj_dir,dataset, round(ptb_rate, 3))
        sp.save_npz(save_dice_dir,sp.csr_matrix(modified_adj))


def generate_PGDAttack_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir, 'PGDAttack', dataset)
    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)
    for i in range(5):
        data = Dataset_MAAM(dataset=dataset)
        print(data.features.shape)
        adj, features, labels = data.adj, data.features, data.labels
        adj,features,labels = preprocess(adj,features,labels,preprocess_adj=False)
        idx_train,idx_val,idx_test = data.idx_train,data.idx_val,data.idx_test
        # set victim conv
        # victim_model = set_surrogate_model(data)
        # Setup Victim Model
        victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                           nhid=16, dropout=0.5, weight_decay=5e-4, device=device).to(device)
        victim_model.fit(features, adj, labels, idx_train)
        # Setup Attack Model
        model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device,
                          attack_structure=True,attack_features=False).to(device)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        ptb_rate = 0.05 * (i + 1)
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(features, adj, labels, idx_train,n_perturbations=n_perturbations)
        modified_adj = model.modified_adj.cpu().numpy()
        # modified_adj_change = modified_adj.detach().numpy()
        # print(modified_adj)
        save_PGDAttack_dir = '{}/PGDAttack_{}_adj_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        # save_PGDAttack_dir = '{}/PGDAttack_{}_features_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        # print(save_PGDAttack_dir)
        sp.save_npz(save_PGDAttack_dir, sp.csr_matrix(modified_adj))
        # break
        # sp.save_npz('{}/PGDAttack/{}/PGDAttack_{}_adj_{}'.format(save_dir,dataset,dataset, round((i + 1) * 0.05, 3)),
        #             sp.csr_matrix(modified_adj))

def generate_MinMax_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir, 'MinMax', dataset)
    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)
    for i in range(5):
        data = Dataset_MAAM(dataset=dataset)
        adj, features, labels = data.adj, data.features, data.labels
        # adj, features, labels变成torch.FloatTensor这个类型
        adj, features, labels = preprocess(adj,features,labels,preprocess_adj=False)
        idx_train,idx_val,idx_test = data.idx_train,data.idx_val,data.idx_test
        # #set victim conv
        victim_model = set_surrogate_model(data)
        # set attack conv
        # 默认情况下，attack_features=False,attack_structure=True
        model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device,
                       attack_features=False,attack_structure=True)
        model = model.to(device)
        ptb_rate = 0.05 * (i + 1)
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj.cpu().numpy()
        # print(modified_adj)
        save_MinMax_dir = '{}/MinMax_{}_adj_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        # save_MinMax_dir = '{}/MinMax_{}_features_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        # print(save_MinMax_dir)
        sp.save_npz(save_MinMax_dir, sp.csr_matrix(modified_adj))
        # break
        # sp.save_npz('{}/MinMax/{}/MinMax_{}_adj_{}'.format(save_dir, dataset, dataset,round((i + 1) * 0.05, 3)),
        #             sp.csr_matrix(modified_adj))

def generate_Metattack_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir, 'Metattack', dataset)
    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)
    # mettack攻击
    for i in range(5):
        data = Dataset_MAAM(dataset=dataset)
        adj, features, labels = data.adj, data.features, data.labels
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_unlabeled = np.union1d(idx_val, idx_test)

        surrogate = set_surrogate_model(data)
        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device,
                          attack_features=False,attack_structure=True)
        model = model.to(device)
        ptb_rate = 0.05 * (i + 1)
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)

        modified_adj = model.modified_adj
        modified_adj_np = sp.csr_matrix(modified_adj.cpu().numpy())

        #save_Metattack_dir = '{}/Metattack_{}_adj_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        # save_Metattack_dir = '{}/Metattack_{}_features_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        # print(save_Metattack_dir)
        #sp.save_npz(save_Metattack_dir, sp.csr_matrix(modified_adj))
        # break
        sp.save_npz('{}/{}/{}/{}_{}_adj_{}'.format(save_dir, 'Metattack',dataset, 'Metattack',dataset,round((i + 1) * 0.05, 3)),
                     modified_adj_np)


if __name__=='__main__':

    adv_dataset_dir = '../'
    # adv_dataset_dir = os.path.abspath(os.path.dirname(adv_dataset_dir))
    # 'cora', 'cora_ml', 'citeseer', 'polblogs',
    dataset_list = ['cora', 'cora_ml', 'citeseer', 'polblogs','pubmed']
    # print(adv_dataset_dir)
    save_adv_data_dir =adv_dataset_dir+'attack_data'

    if not os.path.exists(save_adv_data_dir):
        os.makedirs(save_adv_data_dir)
    for dataset in dataset_list:
        # generate_dice_adj(dataset,save_adv_data_dir)
        # generate_random_adj(dataset,save_adv_data_dir)
        # generate_PGDAttack_adj(dataset,save_adv_data_dir)
        # generate_MinMax_adj(dataset,save_adv_data_dir)
        generate_Metattack_adj(dataset,save_adv_data_dir)
