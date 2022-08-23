from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
import numpy as np
import os.path as osp
import json
import time
from GNNGuard import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='polblogs',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.001, help='pertubation rate')
parser.add_argument('--conv', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'],
                    help='conv variant')

parser.add_argument('--modelname', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard', choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=True, choices=[True, False])


args = parser.parse_args()
attention = args.GNNGuard


def test(data, device):
    # print(data.to)
    # data.cuda()
    # data.to(device)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if type(adj) is torch.Tensor:
        adj = sp.csr_matrix(adj.cpu().numpy())
    ''' testing conv '''
    start = time.perf_counter()
    print(args.modelname)

    classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1,
                                           dropout=0.5, device=device)
    # print(classifier)
    classifier = classifier.to(device)

    # print(type(idx_test))
    # # print(idx_train)
    # # print(idx_val)
    # print(type(labels))
    # print(type(features))
    classifier.fit(features, adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=True, attention=attention)  # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    # torch.save(conv.state_dict(),model_save_dir+'/Median_GCN/'
    #            +arg['attack']+'_'+arg['dataset_name']+'_'+str(arg['i'])
    #            +'_model_'+str(arg['j'])+'.pth')
    classifier.eval()
    end = time.perf_counter()

    print(idx_test)
    acc_test, output = classifier.test(idx_test)
    return acc_test.item()



