import os
import numpy as np
from scipy.sparse import lil_matrix
import csv
import time
import torch
import pandas as pd
from sklearn.metrics import recall_score
from collections import defaultdict
import utils.data_loader

_BEST_CACHE = {}

def calc(n,m,ttuser,ttitem,pre,ttrating,atk=5):
    user=ttuser.cpu().detach().numpy()
    item=ttitem.cpu().detach().numpy()
    pre=pre.cpu().detach().numpy()
    rating=ttrating.cpu().numpy()
    
    # Ground truth positives
    pos_mask = (rating == 1)
    pos_users = user[pos_mask]
    pos_items = item[pos_mask]
    
    # Group positive items by user
    user_pos_items = defaultdict(set)
    for u, i in zip(pos_users, pos_items):
        user_pos_items[u].add(i)
        
    # Predictions
    preall=np.ones((n,m))*(-1000000)
    preall[user,item]=pre
    
    # Top K
    id_sorted = np.argsort(preall, axis=1, kind='quicksort')[:, ::-1]
    top_k_items = id_sorted[:, :atk]
    
    recall_list = []
    ndcg_list = []
    
    # Only evaluate for users with positive items
    for u in user_pos_items:
        ground_truth = user_pos_items[u]
        recommendations = top_k_items[u]
        
        hits = 0
        dcg = 0.0
        idcg = 0.0
        
        for k, item_idx in enumerate(recommendations):
            if item_idx in ground_truth:
                hits += 1
                dcg += 1.0 / np.log2(k + 2)
        
        # IDCG
        num_pos = len(ground_truth)
        min_len = min(num_pos, atk)
        for k in range(min_len):
            idcg += 1.0 / np.log2(k + 2)
            
        recall_list.append(hits / num_pos)
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
        
    return [0, np.mean(recall_list), np.mean(ndcg_list)]

def nll(vector_predict, vector_true):
    return -1 / vector_true.shape[0] * torch.sum(torch.log(1 + torch.exp(-vector_predict * vector_true))).item()

def auc(vector_predict, vector_true): 
    device = vector_predict.device
    pos_indexes = torch.where(vector_true == 1)[0]
    pos_whe=(vector_true == 1)
    sort_indexes = torch.argsort(vector_predict)
    rank=torch.zeros((len(vector_predict)), device=device)
    rank[sort_indexes] = torch.arange(len(vector_predict), dtype=torch.float32, device=device)
    rank = rank * pos_whe
    auc = (torch.sum(rank) - len(pos_indexes) * (len(pos_indexes) - 1) / 2)/ \
            (len(pos_indexes) * (len(vector_predict) - len(pos_indexes)))
    return auc.item()

def mse(vector_predict, vector_true): 
    mse = torch.mean((vector_predict - vector_true)**2)
    return mse.item()

def mae(vector_predict, vector_true): 
    mae = torch.mean(torch.abs(vector_predict - vector_true))
    return mae.item()

def recall_dcg(test_users, test_items, test_ratings, model,top_k, device = 'cuda'): 
  all_user_idx = torch.unique(test_users)
  recall_top_k = []
  dcg_top_k = []
  best_dcg_k = []
  for i in range(len(all_user_idx)):
      index = torch.nonzero(test_users==i)
      user_i = test_users[index].squeeze()
      item_i = test_items[index].squeeze()
      pre_i = model(user_i, item_i)
      y_i = test_ratings[index]
      pre_top_k = torch.argsort(-pre_i)[:top_k]
      y_top_k = y_i[pre_top_k]
      y_true = torch.clamp(y_top_k,min=0)
      count = y_true.sum()
      recall_top_k.append(count.item())
    #   log2_iplus1 = (torch.log2(1+torch.arange(1,top_k+1))).to(device)
    #   dcg = y_true.squeeze()/log2_iplus1
    #   dcg_top_k.append(dcg.sum().item())
    #   best_dcg = y_true.squeeze()[torch.argsort(-y_true.squeeze())][:top_k] / log2_iplus1
    #   best_dcg_k.append(best_dcg.sum().item())

  recall_k = np.mean(recall_top_k)
#   dcg_k = np.mean(dcg_top_k)
  dcg_k = None
#   best_dcg_k = np.mean(best_dcg_k)

#   if np.sum(best_dcg_k) == 0:
#     ndcg_k = 1
#   else:
#     ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

  return recall_k,dcg_k


def recall_func(model, test_users, test_items, y_te, top_k_list = [5, 10]):
    x_te = torch.cat((test_users.reshape(test_users.size()[0],1), test_items.reshape(test_items.size()[0],1)), 1)
    all_user_idx = torch.unique(x_te[:,0]).to('cuda:0')
    all_tr_idx = torch.arange(len(x_te)).to('cuda:0')
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model(x_u[:,0], x_u[:,1])

        for top_k in top_k_list:
            pred_top_k = torch.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()

            temp = torch.sum(y_u)
            if temp == 0:
                recall = 0
                result_map["recall@{}".format(top_k)].append(recall)
            else:
                recall = torch.sum(y_u[pred_top_k]) / temp 
                result_map["recall@{}".format(top_k)].append(recall.item())


    result = {}
    for key, value in result_map.items():
        result[key] = np.mean(value)  
    
    return result


def evaluate(vector_Predict, vector_Test, metric_names, users = None, items = None):
    global_metrics = {
        "AUC": auc,
        "NLL": nll,
        "MSE": mse,
        'Recall_Precision_NDCG@': 5,
        'Recall_Precision_NDCG@10': 10}

    results = {}
    for name in metric_names:
        if name != 'Recall_Precision_NDCG@':
            if name == 'AUC':
                results[name] = auc(vector_predict=vector_Predict, vector_true=vector_Test)
            else:
                results[name] = global_metrics[name](vector_predict=vector_Predict, vector_true=vector_Test)

    if 'Recall_Precision_NDCG@' in metric_names: 
        users_num = torch.max(users).item() + 1
        items_num = torch.max(items).item() + 1


        # yahoo coat
        Recall_Precision_NDCG = calc(users_num, items_num, users, items, vector_Predict, vector_Test, atk=global_metrics['Recall_Precision_NDCG@'])
        results['NDCG@5'] =  Recall_Precision_NDCG[2]
        results['Recall@5'] =  Recall_Precision_NDCG[1]

         
        Recall20, NDCG20 = calc(users_num, items_num, users, items, vector_Predict, vector_Test, atk=20)[1:3]
        results['NDCG@20'] = NDCG20
        results['Recall@20'] = Recall20
        results['RECALL@20'] = Recall20

        Recall50, NDCG50 = calc(users_num, items_num, users, items, vector_Predict, vector_Test, atk=50)[1:3]
        results['NDCG@50'] = NDCG50
        results['Recall@50'] = Recall50
        results['RECALL@50'] = Recall50
        
        
    return results

def upsert_results(method_name, results, dataset_name, filename='results/results.csv'):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    fieldnames = ['Time', 'Method', 'Dataset', 'AUC', 'MSE', 'NDCG@5', 'Recall@5', 'NDCG@20', 'Recall@20', 'NDCG@50', 'Recall@50']
    row = {
        'Time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'Method': method_name,
        'Dataset': dataset_name,
        'AUC': results.get('AUC', ''),
        'MSE': results.get('MSE', ''),
        'NDCG@5': results.get('NDCG@5', ''),
        'Recall@5': results.get('Recall@5', ''),
        'NDCG@20': results.get('NDCG@20', ''),
        'Recall@20': results.get('Recall@20', results.get('RECALL@20', '')),
        'NDCG@50': results.get('NDCG@50', ''),
        'Recall@50': results.get('Recall@50', results.get('RECALL@50', ''))
    }

    rows = []
    if os.path.isfile(filename):
        with open(filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

    replaced = False
    for i, r in enumerate(rows):
        if r.get('Method') == method_name and r.get('Dataset') == dataset_name:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {filename}")

def save_optuna_trial(method_name, dataset_name, trial_number, params, results, filename=None):
    if filename is None:
        filename = f"results/optuna_{method_name}.txt"
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'a') as f:
        f.write(f"Trial {trial_number}\n")
        f.write(f"Dataset: {dataset_name}\n")
        param_str = ', '.join([f'{k}={v:.6f}' if isinstance(v, float) else f'{k}={v}' for k, v in params.items()])
        f.write(f"Params: {param_str}\n")
        for key, val in results.items():
            f.write(f"{key}: {val:.6f}\n")
        f.write('-' * 50 + '\n')

def save_optuna_best(method_name, dataset_name, best_trial_number, best_params, best_results, filename=None):
    if filename is None:
        filename = f"results/optuna_{method_name}.txt"
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'a') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write('BEST TRIAL SUMMARY\n')
        f.write('=' * 80 + '\n')
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Trial number: {best_trial_number}\n")
        param_str = ', '.join([f'{k}={v:.6f}' if isinstance(v, float) else f'{k}={v}' for k, v in best_params.items()])
        f.write(f"Best params: {param_str}\n")
        for key, val in best_results.items():
            f.write(f"{key}: {val:.6f}\n")
        f.write('=' * 80 + '\n')

def update_best_optuna(method_name, dataset_name, trial_number, params, val_results, test_results, filename=None):
    key = (method_name, dataset_name)
    current_auc = val_results.get('AUC', float('-inf'))
    best = _BEST_CACHE.get(key)
    save_optuna_trial(method_name, dataset_name, trial_number, params, val_results, filename=filename)
    if best is None or current_auc > best['auc']:
        _BEST_CACHE[key] = {
            'auc': current_auc,
            'trial_number': trial_number,
            'params': params,
            'test_results': test_results
        }
        upsert_results(method_name, test_results, dataset_name)
        save_optuna_best(method_name, dataset_name, trial_number, params, test_results, filename=filename)
