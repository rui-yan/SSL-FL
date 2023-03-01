# --------------------------------------------------------
# Simulating data splits with different degrees of heterogeneity
# --------------------------------------------------------

import os
import csv
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def non_iid_split_dirichlet(y_train, n_clients, n_classes, beta=0.4):
    '''
    Utility function for data splitting
    Inputs:
        y_train: label of each image (corresponding with image_id) ===> 1-D array
        n_clients: the number of clients in each split
        n_classes: the number of classes in the dataset
        beta: the degree of non-IID based on Dirichlet dist. Smaller beta -> higher heterogeneity
    '''
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
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def split_generator(image_fname, image_label, save_path, n_clients, n_classes, beta_list):
    '''
    Simulate data splits (more general function)
    Inputs: 
        image_fname: image filenames ===> list
        image_label: label of each image (corresponding with image_fname) ===> 1-D array
        save_path: please specify path for saving split files
        n_clients: number of clients
        n_classes: number of classes
        beta_list:  please specify beta for each split
    '''
    for i, beta in enumerate(beta_list):
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
    '''
    Simulate data splits and save to data_path/{n_clients}_clients 
    Use this function if the data was converted to the unified format
    Inputs: 
        data_path is the path where the data is stored
        n_clients is the number of simulated clients
        n_classes is the number of classes in the dataset
        beta_list is the list of betas for different splits (with different non-iid degrees)
    '''
    train_paths = os.path.join(data_path, 'central', 'train.csv')
    train_paths = list({line.strip().split(',')[0] for line in open(train_paths)})
    
    labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
              open(os.path.join(data_path, 'labels.csv'))}
    
    train_labels = {fname:label for fname, label in labels.items() if fname in train_paths}
    
    for split_id, beta in enumerate(beta_list):
        print(f'\n-------split_{split_id+1}-------')
        
        split_path = data_path+f'/{n_clients}_clients'+f'/split_{split_id+1}'
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        
        net_dataidx_map = non_iid_split_dirichlet(np.array(list(train_labels.values())), n_clients, n_classes, beta)
        
        for cid, c_label_idx in net_dataidx_map.items():
            client_split = np.array(list(train_labels.keys()))[c_label_idx]
            client_path = split_path + f'/client_{cid+1}.csv'
            
            print('client_id: ', cid, Counter(np.array(list(train_labels.values()))[c_label_idx]))
            with open(client_path, 'w') as f:
                writer = csv.writer(f, delimiter='\n')
                writer.writerow(client_split)            

    
def view_split(data_path, n_clients, beta_list=[100, 1, 0.5], save_plot=False):
    '''
    Visualize data splits saved in data_path/{n_clients}_clients: 
    '''
    labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
              open(os.path.join(data_path, 'labels.csv'))}
    
    out={}
    for split_id in range(len(beta_list)):
        dist={}
        for k in range(n_clients):
            cur_clint_path = os.path.join(data_path, f'{n_clients}_clients', f'split_{split_id+1}', f'client_{k+1}.csv')
            img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})

            dist[f'client_{k+1}'] = Counter([label for fname, label in labels.items() if fname in img_paths])
        out[f'split_{split_id+1}'] = dist
    
    if save_plot:
        df = pd.DataFrame(out)
        for split_id in range(len(beta_list)):
            df_split = df.iloc[:,split_id].apply(pd.Series)
            df_split = df_split.reindex(sorted(df_split.columns), axis=1)
            df_split['client_id'] = sorted(df_split.index)
            df_split.plot(x='client_id', kind='barh', rot=0, stacked=True, colormap='tab20c', title=f'split{split_id+1}')
            plt.legend(title='class', loc='upper right')
            
            data_set = os.path.split(data_path)[1]
            plt.savefig(f"./plots/{n_clients}_clients/{data_set}_split{split_id+1}.png")
            plt.show()
    
    return out

# data_path='/home/yan/SSL-FL/data/Retina'
# data_split(data_path=data_path, n_clients=5, n_classes=2)
# view_split(data_path, n_clients=5, save_plot=True)
