import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from model import *
import arguments
import utils.load_dataset
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'kuaiRec': 
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [256, 512]}
        args.base_model_args = {'hidden_dim': 32, 'learning_rate': 0.001, 'weight_decay': 5e-3}
        args.propensity_model_args = {'hidden_dim': 32, 'learning_rate': 0.005, 'weight_decay': 1e-5}
        args.imputation_model_args = {'hidden_dim': 32, 'learning_rate': 0.005, 'weight_decay': 1e-5}
    elif args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 4096, 'epochs': 500, 'patience': 60, 'block_batch': [6000, 500]}
        args.base_model_args = {'hidden_dim': 32, 'learning_rate': 0.001, 'weight_decay': 1e-5}
        args.imputation_model_args = {'hidden_dim': 32, 'learning_rate': 0.005, 'weight_decay': 1e-5}
        args.propensity_model_args = {'hidden_dim': 32, 'learning_rate': 0.005, 'weight_decay': 1e-5}
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 256, 'epochs': 500, 'patience': 60, 'block_batch': [64, 64]}
        args.base_model_args = {'hidden_dim': 32, 'learning_rate': 0.001, 'weight_decay': 1e-6}
        args.imputation_model_args = {'hidden_dim': 32, 'learning_rate': 0.001, 'weight_decay': 1e-6}
        args.propensity_model_args = {'hidden_dim': 32, 'learning_rate': 0.001, 'weight_decay': 1e-6}
    else: 
        print('invalid arguments')
        os._exit()

def weighted_quantile(values, weights, q=0.9):
    indices = np.argsort(values)
    sorted_values = values[indices]
    sorted_weights = weights[indices]
    cumulative_weights = np.cumsum(sorted_weights)
    cutoff = q * np.sum(weights)
    idx = np.searchsorted(cumulative_weights, cutoff)
    if idx >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[idx]

def build_sparse_matrix(indices, values, shape, device):
    return torch.sparse_coo_tensor(indices.to(device), values.to(device), torch.Size(shape)).coalesce()

def split_observed(train_data, split_ratio=0.8):
    indices = train_data._indices()
    values = train_data._values()
    nnz = values.size(0)
    perm = torch.randperm(nnz, device=values.device)
    split_point = int(nnz * split_ratio)
    train_idx = perm[:split_point]
    cal_idx = perm[split_point:]
    shape = train_data.shape
    train_mat = build_sparse_matrix(indices[:, train_idx], values[train_idx], shape, values.device)
    cal_mat = build_sparse_matrix(indices[:, cal_idx], values[cal_idx], shape, values.device)
    return train_mat, cal_mat

def train_ips_base_model(train_data, device, dataset_name, base_model_args, propensity_model_args, training_args):
    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    n_user, n_item = train_data.shape
    base_model = MF_basemodel(n_user, n_item, dim=16, dropout=0).to(device)
    propensity_model = MF_basemodel(n_user, n_item, dim=16, dropout=0).to(device)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=base_model_args['learning_rate'], weight_decay=base_model_args['weight_decay'])
    propensity_optimizer = torch.optim.Adam(propensity_model.parameters(), lr=propensity_model_args['learning_rate'], weight_decay=propensity_model_args['weight_decay'])
    bce_none = nn.BCELoss(reduction='none')
    bce_mean = nn.BCELoss(reduction='mean')
    r_matrix = train_data.to_dense()
    observed_matrix = torch.sparse_coo_tensor(train_data._indices(), torch.ones(train_data._values().size()).to(device), r_matrix.size()).to_dense()
    for _ in range(training_args['epochs']):
        for _, users in enumerate(train_loader.User_loader): 
            for _, items in enumerate(train_loader.Item_loader): 
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]
                sub_r = torch.flatten(r_matrix[users].t()[items].t())
                sub_observed = torch.flatten(observed_matrix[users].t()[items].t())
                propensity_model.train()
                p_hat = propensity_model(users_all, items_all).clamp(min=0.05, max=0.95)
                loss_p = bce_mean(p_hat, sub_observed)
                propensity_optimizer.zero_grad()
                loss_p.backward()
                propensity_optimizer.step()
                base_model.train()
                pred_all = base_model(users_all, items_all)
                p_hat = propensity_model(users_all, items_all).clamp(min=0.05, max=0.95)
                inv_prop = 1.0 / p_hat
                e_true = bce_none(pred_all, sub_r)
                loss_base = torch.sum(e_true * inv_prop * sub_observed) / (torch.sum(sub_observed) + 1e-9)
                base_optimizer.zero_grad()
                loss_base.backward()
                base_optimizer.step()
    return base_model, propensity_model

def compute_conformal_threshold(cal_data, f_model, propensity_model, alpha, p_o, training_args, device):
    if cal_data._nnz() == 0:
        return 0.0
    cal_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(cal_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    scores = []
    weights = []
    for _, (users, items, ratings) in enumerate(cal_loader):
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)
        with torch.no_grad():
            f_pred = f_model(users, items)
            p_hat = propensity_model(users, items).clamp(min=0.05, max=0.95)
        r = torch.abs(ratings - f_pred).detach().cpu().numpy()
        w = (p_hat / (1.0 - p_hat)) * ((1.0 - p_o) / p_o)
        w = w.detach().cpu().numpy()
        scores.append(r)
        weights.append(w)
    scores = np.concatenate(scores, axis=0)
    weights = np.concatenate(weights, axis=0)
    q_hat = weighted_quantile(scores, weights, q=(1 - alpha))
    return float(q_hat)

def train_and_eval(train_data, val_data, test_data, device = 'cuda', dataset_name='coat',
        base_model_args: dict = {'hidden_dim': 64, 'learning_rate': 0.01, 'weight_decay': 0.1}, 
        imputation_model_args: dict = {'hidden_dim': 10, 'learning_rate': 0.1, 'weight_decay': 0.1}, 
        training_args: dict =  {'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]},
        propensity_model_args: dict = {'hidden_dim': 4, 'learning_rate': 0.005, 'weight_decay': 0.1},
        alpha: float = 0.5,
        split_ratio: float = 0.8,
        conformal_policy: str = 'hard_reject'):

    train_obs, cal_obs = split_observed(train_data, split_ratio=split_ratio)
    f_model, f_propensity = train_ips_base_model(train_obs, device, dataset_name, base_model_args, propensity_model_args, training_args)
    p_o = train_data._nnz() / (train_data.shape[0] * train_data.shape[1])
    p_o = max(min(p_o, 1 - 1e-6), 1e-6)
    q_hat = compute_conformal_threshold(cal_obs, f_model, f_propensity, alpha, p_o, training_args, device)
    q_hat_tensor = torch.tensor(q_hat, device=device)

    # build data_loader. 
    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)

    r_matrix = train_data.to_dense()
    observed_matrix = torch.sparse_coo_tensor(train_data._indices(), torch.ones(train_data._values().size()).to(device), r_matrix.size()).to_dense()

    n_user, n_item = train_data.shape

    # Use Basemodel which loads pre-trained embeddings internally based on dataset_name
    base_model = MF_basemodel(n_user, n_item, dim=16, dropout=0).to(device)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=base_model_args['learning_rate'], weight_decay=base_model_args['weight_decay'])

    imputation_model = MF_basemodel(n_user, n_item, dim=16, dropout=0).to(device)
    imputation_optimizer = torch.optim.Adam(imputation_model.parameters(), lr=imputation_model_args['learning_rate'], weight_decay=imputation_model_args['weight_decay'])
    
    propensity_model = MF_basemodel(n_user, n_item, dim=16, dropout=0).to(device)
    propensity_optimizer = torch.optim.Adam(propensity_model.parameters(), lr=propensity_model_args['learning_rate'], weight_decay=propensity_model_args['weight_decay'])

    none_criterion = nn.MSELoss(reduction='none')
    criterion = nn.BCELoss(reduction='mean')

    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)
    
    for epo in range(early_stopping.max_epochs):
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                # data
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]

                sub_r = torch.flatten(r_matrix[users].t()[items].t())
                sub_observed = torch.flatten(observed_matrix[users].t()[items].t())


                propensity_model.train()
                p_hat = propensity_model(users_all, items_all).clamp(min=0.05)
                loss_p = criterion(p_hat, sub_observed) # P(O=1) learning
                
                propensity_optimizer.zero_grad()
                loss_p.backward()
                propensity_optimizer.step()

                base_model.train()
                imputation_model.eval() 

                p_hat_all = propensity_model(users_all, items_all).clamp(min=0.05).detach()
                inv_prop = 1.0 / p_hat_all
                
                pred_all = base_model(users_all, items_all)
                with torch.no_grad():
                    imp_all = imputation_model(users_all, items_all)
                    f_pred = f_model(users_all, items_all)

                delta_imp = imp_all - f_pred
                if conformal_policy == 'hard_reject':
                    mask_trusted = (1.0 - sub_observed) * (torch.abs(delta_imp) <= q_hat_tensor).float()
                    mask_oi = sub_observed + mask_trusted
                    imp_for_direct = imp_all
                elif conformal_policy == 'clip':
                    imp_for_direct = f_pred + torch.clamp(delta_imp, min=-q_hat_tensor, max=q_hat_tensor)
                    mask_oi = torch.ones_like(sub_observed)
                elif conformal_policy == 'fpred':
                    imp_for_direct = torch.where(torch.abs(delta_imp) <= q_hat_tensor, imp_all, f_pred)
                    mask_oi = torch.ones_like(sub_observed)
                else:
                    raise ValueError(f'Unknown conformal_policy: {conformal_policy}')
                
                bce_none = nn.BCELoss(reduction='none')
                e_true = bce_none(pred_all, sub_r) # Loss against ground truth
                
                e_imp_term = none_criterion(pred_all, imp_all)
                e_imp_term_direct = none_criterion(pred_all, imp_for_direct)
                
                ips_element = (e_true - e_imp_term) * inv_prop
                ips_loss = torch.sum(ips_element * sub_observed) / (torch.sum(sub_observed) + 1e-9)
                
                direct_loss = torch.sum(e_imp_term_direct * mask_oi) / (torch.sum(mask_oi) + 1e-9)
                
                loss_base = ips_loss + direct_loss
                
                base_optimizer.zero_grad()
                loss_base.backward()
                base_optimizer.step()
                
                imputation_model.train()
                base_model.eval() # Fix base model
                
                with torch.no_grad():
                    pred_fixed = base_model(users_all, items_all)
                    p_hat_fixed = propensity_model(users_all, items_all).clamp(min=0.05)
                    inv_prop_fixed = 1.0 / p_hat_fixed
                    
                    # e_loss (prediction error)
                    e_loss = bce_none(pred_fixed, sub_r)
                   
                imp_out = imputation_model(users_all, items_all)
                e_hat_loss = bce_none(imp_out, pred_fixed)
                
                # Loss = ((e_loss - e_hat_loss)^2 * inv_prop) * observed
                sq_diff = (e_loss - e_hat_loss) ** 2
                loss_imp_var = torch.sum(sq_diff * inv_prop_fixed * sub_observed) / (torch.sum(sub_observed) + 1e-9)
                
                imputation_optimizer.zero_grad()
                loss_imp_var.backward()
                imputation_optimizer.step()

        # Validation
        base_model.eval()
        with torch.no_grad():
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    pre_ratings = base_model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))

            val_pre_ratings = torch.empty(0).to(device)
            val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = base_model(users, items)
                val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                val_ratings = torch.cat((val_ratings, ratings))
            
        train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])

        print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), 
                ' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))

        if early_stopping.check([val_results['AUC']], epo):
            break

    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    base_model.load_state_dict(early_stopping.best_state)

    # Test
    test_users = torch.empty(0, dtype=torch.int64).to(device)
    test_items = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings = torch.empty(0).to(device)
    test_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(test_loader):
        pre_ratings = base_model(users, items)
        test_users = torch.cat((test_users, users))
        test_items = torch.cat((test_items, items))
        test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
        test_ratings = torch.cat((test_ratings, ratings))
    
    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])
    test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items)

    print('-'*30)
    print('The performance of validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))
    print('The performance of testing set: {}'.format(' '.join([key+':'+'%.3f'%test_results[key] for key in test_results])))
    print('-'*30)

    return val_results,test_results

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, validation, test = utils.load_dataset.load_dataset0(data_name=args.dataset, type = 'explicit', seed = args.seed, device=device)

    def objective(trial):
        dim = trial.suggest_categorical('dim', [8, 16, 32, 64])
        base_model_args = {
            'hidden_dim': dim,
            'learning_rate': trial.suggest_categorical('base_lr', [0.001, 0.005, 0.01, 0.02, 0.05]),
            'weight_decay': trial.suggest_float('base_wd', 1e-6, 5e-3, log=True)
        }
        imputation_model_args = {
            'hidden_dim': dim,
            'learning_rate': trial.suggest_categorical('imp_lr', [0.001, 0.005, 0.01, 0.02, 0.05]),
            'weight_decay': trial.suggest_float('imp_wd', 1e-6, 5e-3, log=True)
        }
        propensity_model_args = {
            'hidden_dim': dim,
            'learning_rate': trial.suggest_categorical('prop_lr', [0.001, 0.005, 0.01, 0.02, 0.05]),
            'weight_decay': trial.suggest_float('prop_wd', 1e-6, 5e-3, log=True)
        }

        alpha = trial.suggest_float('alpha', 0.05, 0.5, step=0.05)
        split_ratio = trial.suggest_float('split_ratio', 0.5, 0.9, step=0.05)

        val_results, test_results = train_and_eval(
            train, validation, test, device,
            base_model_args=base_model_args,
            imputation_model_args=imputation_model_args,
            training_args=args.training_args,
            propensity_model_args=propensity_model_args,
            dataset_name=args.dataset,
            alpha=alpha,
            split_ratio=split_ratio,
            conformal_policy=args.conformal_policy
        )
        utils.metrics.update_best_optuna(
            f'Conformal_DR-JL-{args.conformal_policy}',
            args.dataset,
            trial.number,
            {**trial.params, 'conformal_policy': args.conformal_policy},
            val_results,
            test_results,
            filename=f'results/optuna_Conformal_DR-JL-{args.conformal_policy}-{args.dataset}.txt'
        )
        return val_results['AUC']

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('-'*30)
    print('Best Trial:', study.best_trial.number)
    print('Best AUC:', study.best_trial.value)
    print('Best Params:', study.best_trial.params)
