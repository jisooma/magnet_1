# _*_codeing=utf-8_*_
# @Time:2022/7/21  19:00
# @Author:mazhixiu
# @File:test_defense.py

from defense import *
defense_dict = {
                'GCN':'GCN_',
                # 'GAT':'GAT_',
                # 'RGCN':'R_GCN',
                # 'GCNSVD':'GCN_SVD',
                # 'GCNJaccard':'GCN_Jaccard',
                # 'ProGCN':'Pro_GCN',
                # 'GNNGuard':'GNN_Guard',
                # 'MedianGCN': 'Median_GCN',
                # 'ElasticGCN': 'Elastic_GCN',
                # 'SimpGCN':'Simp_GCN',
                }
print(device)
def test():

    import random
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print()
    save_dir = './results/adv/'

    for defense, function in defense_dict.items():

        path_1 = save_dir + '/acc'
        path_2 = save_dir + '/var'
        path_3 = save_dir + '/time'
        # 'FGA',
        judge_dir(path_1)
        judge_dir(path_2)
        judge_dir(path_3)
        #
        for dataset in ['cora','cora_ml','citeseer','polblogs','pubmed']:
            for attack in ['Nettack','RND','SGAttack']:
                var_list = []
                acc_list = []
                time_list = []
                for ptb in [1.0,2.0,3.0,4.0,5.0]:
                    data = Dataset_MAAM(dataset=dataset,attack=attack,ptb=ptb)
                    # data.to(device)
                    # print(data.features)

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

                        output,times = eval(function)(data, device, arg)
                        all_time = all_time+times
                        sum = sum + output
                        var.append(output)

                    acc_var = np.std(var)
                    mean_acc = sum / epoch
                    mean_time = all_time/epoch
                    acc_list.append(mean_acc)
                    var_list.append(acc_var)
                    time_list.append(mean_time)

                print("acc_list:", acc_list)
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc',  defense,attack,dataset), np.array(acc_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var',  defense,attack,dataset), np.array(var_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/time', defense,attack,dataset), np.array(time_list))

test()