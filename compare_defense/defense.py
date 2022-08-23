# _*_codeing=utf-8_*_
# @Time:2022/4/10  12:05
# @Author:mazhixiu
# @File:defense.py
import os
import sys
sys.path.append('/home/mzx/MAAM_version_3')
# from torch_geometric.nn import GCN
from deeprobust.graph.defense import GCN, GAT, RGCN, GCNSVD, GCNJaccard, SimPGCN, MedianGCN, ProGNN
from elastic_gnn import ElasticGNN
import torch_geometric.transforms as T
from test_GNNGuard import *
from utils.utils import *


def GCN_(data, device, arg):
    # Setup GCN Model
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    model = GCN(nfeat=features.shape[1], nhid=128, nclass=labels.max() + 1, device=device,lr=0.01)
    model = model.to(device)
    # # using validation to pick conv
    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=500, verbose=False,with_bias=False)
    end = time.perf_counter()
    model.eval()
    output = model.test(idx_test)
    return output,end-start


def GAT_(data, device, arg):
    # start = time.perf_counter()
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    pyg_data = Dpr2Pyg(data).pyg_data
    # print(pyg_data[0])
    gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device,lr=0.01)
    gat = gat.to(device)
    start = time.perf_counter()
    gat.fit(pyg_data, train_iters=500,verbose=False)  # train with earlystopping
    end = time.perf_counter()
    output = gat.test()
    return output,end-start


def GCN_SVD(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
                   nhid=128, device=device,lr=0.01,with_bias=False)

    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val,train_iters=500,k=arg, verbose=False)
    end = time.perf_counter()

    model.eval()
    output = model.test(idx_test)
    return output,end-start

def GCN_Jaccard(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1,
                       nhid=128, device=device,lr=0.01,with_bias=False)

    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val,train_iters=500,k=arg, verbose=False)
    end = time.perf_counter()

    model.eval()
    output = model.test(idx_test)
    return output,start-end

def R_GCN(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                 nhid=128, device=device,lr=0.01)
    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=500, verbose=False)
    end = time.perf_counter()
    output = model.test(idx_test)
    return output,time.perf_counter()-start


def GNN_Guard(data, device, arg):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if type(adj) is torch.Tensor:
        adj = sp.csr_matrix(adj.cpu().numpy())
    ''' testing conv '''

    # print(args.modelname)

    model = GNNGuard(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1,
                     dropout=0.5, device=device,lr=0.01)
    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, train_iters=500,
              idx_val=idx_val,
              idx_test=idx_test,
              verbose=False, attention=attention)  # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    end = time.perf_counter()
    model.eval()
    output = model.test(idx_test)
    return output,end-start


def Pro_GCN(data, device, arg):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, device=device)

    start = time.perf_counter()
    perturbed_adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)
    prognn = ProGNN(model, args, device)
    prognn.fit(features, perturbed_adj, labels, idx_train, idx_val)
    end = time.perf_counter()
    output = prognn.test(features, labels, idx_test)
    return output,end-start


def Median_GCN(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    pyg_data = Dpr2Pyg(data).pyg_data

    model = MedianGCN(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1,
                      dropout=0.5, device=device,lr=0.01)
    model = model.to(device)

    start = time.perf_counter()
    model.fit(pyg_data=pyg_data,train_iters=500, verbose=False)
    end = time.perf_counter()

    model.eval()
    output = model.test(pyg_data)
    return output,end-start


def Simp_GCN(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1], nhid=128,
                    nclass=labels.max() + 1, device=device,lr=0.01)
    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=500, verbose=False)
    end = time.perf_counter()

    output = model.test(idx_test)
    return output,end-start


def Elastic_GCN(data, device, arg):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = ElasticGNN(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1, device=device,lr=0.01)
    model = model.to(device)
    pyg_data = Dpr2Pyg(data)

    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    data1 = transform(pyg_data.data)

    start = time.perf_counter()
    model.fit(data1,train_iters=500)  # train with earlystopping
    end = time.perf_counter()

    model.eval()
    output = model.test(data1)
    return output,end-start


def judge_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        os.makedirs(path)

def test_1():
    save_dir = './results/clean/test_new_1'
    defense_dict = {
        'GCN': 'GCN_',
        # 'GAT': 'GAT_',
        # 'RGCN': 'R_GCN',
        # 'GCNSVD': 'GCN_SVD',
        # 'GCNJaccard': 'GCN_Jaccard',
        # # 'ProGCN':'Pro_GCN',
        # 'GNNGuard': 'GNN_Guard',
        # 'MedianGCN': 'Median_GCN',
        # 'ElasticGCN': 'Elastic_GCN',
        # 'SimpGCN': 'Simp_GCN',
    }
    for defense, function in defense_dict.items():
        var_list = []
        acc_list = []
        time_list = []
        path_1 = save_dir + '/acc/'
        path_2 = save_dir + '/var/'
        path_3 = save_dir + '/time/'

        judge_dir(path_1)
        judge_dir(path_2)
        judge_dir(path_3)

        for dataset in dataset_list:
            data = Dataset_MAAM(dataset=dataset)
            epoch = 10
            sum = 0
            all_time = 0
            var = []
            for i in range(epoch):
                print("defense:", defense)
                arg = None
                if defense == 'Jaccard':
                    arg = random.randint(1, 5) * 0.02
                if defense == 'GCNSVD':
                    k_arr = [5, 10, 15, 50, 100, 200]
                    arg = k_arr[random.randint(0, 5)]

                output, times = eval(function)(data, device, arg)
                all_time = all_time + times
                sum = sum + output
                var.append(output)

            acc_var = np.std(var)
            mean_acc = sum / epoch
            mean_time = all_time / epoch
            acc_list.append(mean_acc)
            var_list.append(acc_var)
            time_list.append(mean_time)

        print("acc_list:", acc_list)
        print("var_list:", var_list)
        print('time_list:', time_list)
        np.savetxt("{}/{}_{}.txt".format(save_dir + '/acc', 'clean', defense), np.array(acc_list))
        np.savetxt("{}/{}_{}.txt".format(save_dir + '/var', 'clean', defense), np.array(var_list))
        np.savetxt("{}/{}_{}.txt".format(save_dir + '/time', 'clean', defense), np.array(time_list))

def test():

    import random
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print()
    save_dir = './results/adv/'
    # 防御列表
    defense_dict = {
        'GCN': 'GCN_',
        'GAT': 'GAT_',
        'RGCN': 'R_GCN',
        'GCNSVD': 'GCN_SVD',
        'GCNJaccard': 'GCN_Jaccard',
        # 'ProGCN':'Pro_GCN',
        'GNNGuard': 'GNN_Guard',
        'MedianGCN': 'Median_GCN',
        'ElasticGCN': 'Elastic_GCN',
        'SimpGCN': 'Simp_GCN',
    }
    for defense, function in defense_dict.items():

        path_1 = save_dir + '/acc'
        path_2 = save_dir + '/var'
        path_3 = save_dir + '/time'
        # 'FGA',
        judge_dir(path_1)
        judge_dir(path_2)
        judge_dir(path_3)
        #
        for dataset in ['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed']:
            # for attack in ['Nettack', 'RND', 'SGAttack']:
            for attack in ['Metattack']:
                var_list = []
                acc_list = []
                time_list = []
                # for ptb in [1.0, 2.0, 3.0, 4.0, 5.0]:
                for ptb in [0.05,0.10,0.15,0.20,0.25]:
                    data = Dataset_MAAM(dataset=dataset, attack=attack, ptb=ptb)

                    epoch = 10
                    sum = 0
                    all_time = 0
                    var = []
                    for i in range(epoch):
                        print("defense:", defense)
                        arg = None
                        if defense == 'Jaccard':
                            arg = random.randint(1, 5) * 0.02
                        if defense == 'GCNSVD':
                            k_arr = [5, 10, 15, 50, 100, 200]
                            arg = k_arr[random.randint(0, 5)]

                        output, times = eval(function)(data, device, arg)
                        all_time = all_time + times
                        sum = sum + output
                        var.append(output)

                    acc_var = np.std(var)
                    mean_acc = sum / epoch
                    mean_time = all_time / epoch
                    acc_list.append(mean_acc)
                    var_list.append(acc_var)
                    time_list.append(mean_time)

                print("acc_list:", acc_list)
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc', defense, attack, dataset),
                           np.array(acc_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var', defense, attack, dataset),
                           np.array(var_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/time', defense, attack, dataset),
                           np.array(time_list))

if __name__=="__main__":
    import random
    test_1()

"""
data->denfese
"""

# data = load_Dataset('Cora')
# GNN_Guard(data=data,device=device,k='1')
