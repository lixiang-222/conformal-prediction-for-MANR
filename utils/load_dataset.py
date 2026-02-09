import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch.utils.data import Dataset, DataLoader
import torch


def seed_randomly_split(df, ratio, split_seed, shape):

    np.random.seed(split_seed)

    rows, cols, rating = df['uid'], df['iid'], df['rating']
    num_nonzeros = len(rows)
    permute_indices = np.random.permutation(num_nonzeros)
    rows, cols, rating = rows.iloc[permute_indices], cols.iloc[permute_indices], rating.iloc[permute_indices]

    idx = int(ratio[0] * num_nonzeros)

    valid = sp.csr_matrix((rating.iloc[:idx], (rows.iloc[:idx], cols.iloc[:idx])),
                          shape=shape, dtype='float32')
    test = sp.csr_matrix((rating.iloc[idx:], (rows.iloc[idx:], cols.iloc[idx:])),
                         shape=shape, dtype='float32')

    return valid, test


def load_dataset0(data_name='coat', type = 'explicit', seed=0, threshold=4, device = 'cuda'): 
    if type not in ['explicit', 'implicit', 'list']: 
        print('--------illegal type, please reset legal type.---------')
        return
    path = os.path.dirname(os.path.dirname(__file__)) + '/datasets/' + data_name
    if data_name == 'simulation': 
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=None, names=['uid', 'iid', 'position', 'rating'])
    else: 
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=None, names=['uid', 'iid', 'rating'])
    random_df = pd.read_csv(path + '/random.txt', sep=',', header=None, names=['uid', 'iid', 'rating'])

    if type == 'implicit': 
        user_df = user_df.drop(user_df[user_df['rating'] < threshold].index)


    user_df_copy = user_df.copy()
    if data_name == "kuaiRec":
        threshold = 2
        user_df['uid'] = pd.factorize(user_df['uid'])[0] 
        user_df['iid'] = pd.factorize(user_df['iid'])[0] 
        uid_mapping = dict(zip(user_df_copy['uid'], user_df['uid']))
        iid_mapping = dict(zip(user_df_copy['iid'], user_df['iid']))
        random_df['uid'] = random_df['uid'].map(uid_mapping)
        random_df['iid'] = random_df['iid'].map(iid_mapping)


    user_df.loc[user_df['rating'] < threshold, 'rating'] = 0
    user_df.loc[user_df['rating'] >= threshold, 'rating'] = 1

    random_df.loc[random_df['rating'] < threshold, 'rating'] = 0
    random_df.loc[random_df['rating'] >= threshold, 'rating'] = 1


    m, n = max(user_df['uid']) + 1, max(user_df['iid']) + 1
    ratio = (0.05, 0.95)
    validation, test = seed_randomly_split(df=random_df, ratio=ratio, split_seed=seed, shape=(m, n))

    if type == 'list': 
        train_pos = sp.csr_matrix((user_df['position'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='int64')
        train_rating = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')
        train = {}
        train['position'] = sparse_mx_to_torch_sparse_tensor(train_pos).to(device)
        train['rating'] = sparse_mx_to_torch_sparse_tensor(train_rating).to(device)
    else: 
        train = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')
        train = sparse_mx_to_torch_sparse_tensor(train).to(device)

    validation = sparse_mx_to_torch_sparse_tensor(validation).to(device)
    test = sparse_mx_to_torch_sparse_tensor(test).to(device)

    return train, validation, test


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
