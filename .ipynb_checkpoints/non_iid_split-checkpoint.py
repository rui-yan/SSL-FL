import numpy as np
import torch
import csv
import os
import pandas as pd
def non_iid_split_dirichlet(y_train, n_parties, K, beta=0.4):

    min_size = 0
    min_require_size = 10
    #K = 10
    #if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
    #    K = 2
    #    # min_require_size = 100

    N = y_train.shape[0]
    np.random.seed(2020)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


# def split_generator(image_id, image_label, savepath, n_parties, K, betaall):
#     '''
#     image_id: idx of each image ===> list
#     image_label: label of each image (corresponding with image_id) ===> 1-D array
#     savepath: please specify path for saving split files
#     n_parties: number of clients
#     K: number of classes
#     betaall:  please specify beta for each split
#     '''
#     count = 1
#     client_split_info = {}
#     for beta in betaall:
#         net_dataidx_map = non_iid_split_dirichlet(image_label, n_parties, K, beta)
#         headers = []
#         for i in range(n_parties):
#             headers.append('client' + str(i+1))
#             client_split = [image_id[x] for x in net_dataidx_map[i]]
#             client_split_info['client' + str(i+1)] = client_split
#         cvs_savepath = savepath + '/split' + str(count) + '.csv'
#         with open(cvs_savepath, 'w', newline='') as f:
#             # 标头在这里传入，作为第一行数据
#             writer = csv.DictWriter(f, headers)
#             writer.writeheader()
#             writer.writerow(client_split_info)
#         count = count + 1


def split_generator(image_fname, image_label, save_path, n_parties, K, betaall):
    '''
    image_id: idx of each image ===> list
    image_label: label of each image (corresponding with image_id) ===> 1-D array
    savepath: please specify path for saving split files
    n_parties: number of clients
    K: number of classes
    betaall:  please specify beta for each split
    '''
    for i, beta in enumerate(betaall):
        split_path = save_path + f'/split_{i+1}'
        if not os.path.exists(split_path):
            os.makedirs(split_path)
            
        net_dataidx_map = non_iid_split_dirichlet(image_label, n_parties, K, beta)
        
        for k in range(n_parties):
            client_split = [image_fname[x] for x in net_dataidx_map[k]]
            client_path = split_path + f'/client_{k+1}.csv'
            with open(client_path, 'w') as f:
                writer = csv.writer(f, delimiter='\n')
                writer.writerow(client_split)


# y_train = np.zeros([5000])
# y_train[:1000] = 1
# y_train[1000:1150] = 2
# y_idx = [str(i) + '.img' for i in range(5000)]

# n_parties = 5

# split_generator(y_path, y_label, r'./', n_parties, 3, [100, 1.0, 0.5])


'''
def non_iid_split(y_train, n_parties, K, amount_ratio, label_ratio, beta=0.4):
    ''
    y_train:
    n_parties: number of clients
    K: number of class
    amount_ratio: ratio of the total number of patients in each client
                  > > > array shape n,
                  > > > eg. [0.25, 0.3, 0.45]
                  > > > sum has to be 1
    label_ratio: ratio of each label in each client
                  > > > array shape n, K
                  > > > [[0.25, 0.3, 0.45]，
                          [0.1, 0.2, 0.2],
                          [0.65, 0.5, 0.35]]
    ''

    assert amount_ratio.sum() != 1, 'sum of amount ratio is not equal to zero!'
    assert amount_ratio.shape != n_parties, 'amount ratio is not corresponding to the amount of client!'
    assert label_ratio.shape != [n_parties, K], 'label ratio is not corresponding to the amount of Client or Class!'
    #for i in range(K):
    #    assert label_ratio[:, i].sum() != 1.0, 'sum of ratio of Class' + str(i) + ' is not equal to zero!'

    np.random.seed(2022)
    net_dataidx_map = {}

    idx_batch = [[] for _ in range(n_parties)]
    for k in range(K):
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)
        num_k = idx_k.shape[0]
        k_ratio = label_ratio[:, k]
        seg = num_k * k_ratio
        for n in range(n_parties):
            if n == 0:
                st = 0
                en = st + int(seg[n])
            else:
                if n < n_parties - 1:
                    st = en
                    en = st + int(seg[n])
                else:
                    st = en
                    en = num_k
            idx_batch[n].extend(idx_k[st:en].tolist())

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map
'''
