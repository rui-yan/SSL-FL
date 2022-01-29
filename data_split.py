import os
import csv
import torch
from collections import Counter
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def non_iid_split_dirichlet(y_train, n_clients, n_classes, beta=0.4):
    min_size = 0
    min_require_size = 10

    N = y_train.shape[0]
    np.random.seed(2022)
    net_dataidx_map = {}
    
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_clients))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if n_classes == 2 and n_clients <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def split_generator(image_fname, image_label, save_path, n_clients, n_classes, betaall):
    '''
    image_id: idx of each image ===> list
    image_label: label of each image (corresponding with image_id) ===> 1-D array
    savepath: please specify path for saving split files
    n_clients: number of clients
    n_classes: number of classes
    betaall:  please specify beta for each split
    '''
    for i, beta in enumerate(betaall):
        split_path = save_path + f'/split_{i+1}'
        if not os.path.exists(split_path):
            os.makedirs(split_path)
            
        net_dataidx_map = non_iid_split_dirichlet(image_label, n_clients, n_classes, beta)
        
        for k in range(n_clients):
            client_split = [image_fname[x] for x in net_dataidx_map[k]]
            client_path = split_path + f'/client_{k+1}.csv'
            with open(client_path, 'w') as f:
                writer = csv.writer(f, delimiter='\n')
                writer.writerow(client_split)
                

def data_split(data_path, n_clients, n_classes, beta_list=[100, 1, 0.5]):
    train_paths = os.path.join(data_path, 'central', 'train.csv')
    train_paths = list({line.strip().split(',')[0] for line in open(train_paths)})
    
    labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
              open(os.path.join(data_path, 'labels.csv'))}
    
    train_labels = {fname:label for fname, label in labels.items() if fname in train_paths}
    # train_labels = np.array([label for fname, label in labels.items() if fname in train_paths])
    # print('data_splits in train dataset: ', Counter(train_labels)) # {0.0: 4500, 1.0: 4499}
    
    print(len(train_paths))
    print(len(train_labels))
    # split_generator(fname, label, r'/raid/yan/COVIDx', 5, 3, [100, 1.0, 0.5])
    
    for split_id, beta in enumerate(beta_list):
        print(f'\n-------split_{split_id+1}-------')
        
        split_path = data_path + f'/split_{split_id+1}'
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        
        # print(train_labels.values())
        net_dataidx_map = non_iid_split_dirichlet(np.array(list(train_labels.values())), n_clients, n_classes, beta)
        
        for cid, c_label_idx in net_dataidx_map.items():
            client_split = np.array(list(train_labels.keys()))[c_label_idx]
            client_path = split_path + f'/client_{cid+1}.csv'
            
            print('client_id: ', cid, Counter(np.array(list(train_labels.values()))[c_label_idx]))
            # print(client_split[:5])
            with open(client_path, 'w') as f:
                writer = csv.writer(f, delimiter='\n')
                writer.writerow(client_split)            
    
    # print([{client_id:len(paths)} for client_id, paths in net_dataidx_map.items()])

    
def view_split(data_path, n_clients, save_plot=False):
    
    labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
              open(os.path.join(data_path, 'labels.csv'))}
    
    out={}
    for split_id in range(3):
        dist={}
        for k in range(n_clients):
            cur_clint_path = os.path.join(data_path, f'split_{split_id+1}', f'client_{k+1}.csv')
            img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
            # print(img_paths[:5])
            dist[f'client_{k+1}'] = Counter([label for fname, label in labels.items() if fname in img_paths])
        out[f'split_{split_id+1}'] = dist
        # print(f'split_{split_id+1}: ', dist)
    
    if save_plot:
        df = pd.DataFrame(out)
        for split_id in range(3):
            df_split = df.iloc[:,split_id].apply(pd.Series)
            df_split = df_split.reindex(sorted(df_split.columns), axis=1)
            df_split['client_id'] = sorted(df_split.index)
            df_split.plot(x='client_id', kind='barh', rot=0, stacked=True, colormap='tab20c', title=f'split{split_id+1}')
            plt.legend(title='class', loc='upper right')
            
            data_set = os.path.split(data_path)[1]

            plt.savefig(f"/home/yan/SSL-FL/plots/{data_set}_split{split_id+1}.png")
            plt.show()
    
    return out

data_path='/data/yan/SSL-FL/Retina'
# data_path='/data/yan/SSL-FL/COVIDx'
# data_split(data_path, 5, 2)
view_split(data_path, 5, save_plot=True)

# Test Retina splits
