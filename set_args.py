import torch

import numpy as np
import os
workspace = os.getcwd()

path = os.path.abspath(os.path.dirname(__file__))

# 数据集列表
#[,,'Cora'
dataset_list = ['cora','cora_ml','citeseer', 'polblogs','pubmed']
# 攻击列表:'SGAttack','IGAttack',
attack_list = ['Nettack','FGA','Random','DICE','PGDAttack','MinMax','Metattack']
# 防御列表
defense_list = ['GCN','GAT','RGCN','GCNSVD','GCNJaccard', 'GNNGuard','ProGCN','SimpGCN','MedianGCN','ElasticGCN']
# 定向攻击列表
# ,'Nettack','FGA',,'SGAttack',
targeted_attack_list = ['IGAttack']
# global_attack_list = ['Random','Dice','PGDAttack','MinMax']
# 全局攻击列表
global_attack_list = ['Random','DICE','PGDAttack','MinMax','Metattack']

# pyg_dataset_list=['cora','cora_ml','CiteSeer','Pubmed']
LOADERS = {'cora_ml': 'CitationFull',
            'cora': 'CoraFull',
            'Cora': 'Planetoid',
            'Citeseer': 'Planetoid',
            'PubMed': 'Planetoid',
            'polblogs': 'Polblogs',
            # 'CS': 'Coauthor',
            # 'Physics': 'Coauthor',
            # 'Photo': 'Amazon',
            # 'Computers': 'Amazon',
            # 'DBLP': 'CitationFull'
           }
# 干净数据集保存路径
clean_dataset_dir=path+'/clean_data/'
print(clean_dataset_dir)
# 对抗样本保存路径
adv_dataset_dir=path+'/attack_data'
# 对比实验数据保存路径
experiment_data_dir = path+'/experiment_data'
# 对比试验图表保存路径
experiment_figure_dir = path + '/experiment_figure'
# 模型保存路径
model_save_dir = path+'/model_save'

# 防御列表
defense_dict = {
                'GCN':'GCN_',
                'GAT':'GAT_',
                'RGCN':'R_GCN',
                'GCNSVD':'GCN_SVD',
                'GCNJaccard':'GCN_Jaccard',
                # 'ProGCN':'Pro_GCN',
                'GNNGuard':'GNN_Guard',
                'MedianGCN': 'Median_GCN',
                'ElasticGCN': 'Elastic_GCN',
                'SimpGCN':'Simp_GCN',
                }


# parser.add_argument('--seed', type=int, default=15,help='Random seed')
cuda = torch.cuda.is_available()
device = torch.device('cuda:1'if torch.cuda.is_available() else 'cpu')

seed=15
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)




