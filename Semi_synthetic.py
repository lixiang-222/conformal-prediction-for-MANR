import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from model import MLP_basemodel
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class SimpleEarlyStopping:
    def __init__(self, patience=60, max_epochs=500):
        self.patience = patience
        self.max_epochs = max_epochs
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
    
    def check(self, metric, epoch, model):
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_state = model.state_dict().copy()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience



def save_results(method_name, trial_number, params, results, filename):
    os.makedirs('results', exist_ok=True)
    with open(filename, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Trial: {trial_number}\n")
        f.write(f"Params: {params}\n")
        f.write(f"Results: {results}\n")
        f.write(f"{'='*60}\n")

def load_predicted_matrix(filepath="data/predicted_matrix"):
    with open(filepath, "rb") as f:
        prediction = pickle.load(f)
        user_num = pickle.load(f)
        item_num = pickle.load(f)
    return prediction, user_num, item_num

def generate_semisynthetic_data(prediction, p_base=0.5, propensity_type='mnar'):
    total_num = prediction.shape[0]
    ground_truth = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    
    if propensity_type == 'mnar':
        propensity = np.ones_like(ground_truth) * p_base
        propensity[ground_truth <= 0.2] = p_base ** 4.0 
        propensity[(ground_truth > 0.2) & (ground_truth <= 0.4)] = p_base ** 4.0
        propensity[(ground_truth > 0.4) & (ground_truth <= 0.6)] = p_base ** 3.0
        propensity[(ground_truth > 0.6) & (ground_truth <= 0.8)] = p_base ** 2.0
        propensity[ground_truth > 0.8] = p_base ** 1.0
        
    elif propensity_type == 'ground-truth':
        propensity = np.maximum(ground_truth, 0.05)
        
    else:
        raise ValueError(f"Unknown propensity_type: {propensity_type}. "
                        f"Choose from: 'mnar', 'ground-truth'")
    
    observation = np.random.binomial(1, propensity)
    y = ground_truth
    
    return ground_truth, propensity, observation, y

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

def train_ips_reference_model(train_data, propensity_matrix, device, n_user, n_item, 
                             base_model_args, training_args):
    f_model = MLP_basemodel(n_user, n_item, 
                           hidden_dim=base_model_args['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(f_model.parameters(), lr=base_model_args['learning_rate'], 
                                weight_decay=base_model_args['weight_decay'])
    
    mse_none = nn.MSELoss(reduction='none')
    r_matrix = train_data.to_dense()
    observed_matrix = torch.sparse_coo_tensor(train_data._indices(), 
                                             torch.ones(train_data._values().size()).to(device), 
                                             r_matrix.size()).to_dense()
    
    batch_size = training_args['block_batch'][0]
    for _ in range(training_args['epochs']):
        perm = torch.randperm(n_user, device=device)
        for i in range(0, n_user, batch_size):
            users = perm[i:min(i+batch_size, n_user)]
            items = torch.randint(0, n_item, (len(users) * 10,), device=device) % n_item
            
            users_all = users.repeat_interleave(10)
            items_all = items[:len(users_all)]
            
            sub_r = r_matrix[users_all, items_all]
            sub_observed = observed_matrix[users_all, items_all]
            sub_prop = propensity_matrix[users_all, items_all].clamp(min=0.05, max=0.95)
            
            f_model.train()
            pred = f_model(users_all, items_all)
            inv_prop = 1/2 * (1.0 / sub_prop) + 1/2
            e_loss = mse_none(pred, sub_r)
            loss = torch.sum(e_loss * inv_prop * sub_observed) / (torch.sum(sub_observed) + 1e-9)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return f_model

def compute_conformal_threshold(cal_data, f_model, propensity_matrix, alpha, p_o, device, verbose=False):
    if cal_data._nnz() == 0:
        return 0.0, {}
    
    indices = cal_data._indices()
    values = cal_data._values()
    users = indices[0]
    items = indices[1]
    ratings = values
    
    with torch.no_grad():
        f_pred = f_model(users, items)
        p_hat = propensity_matrix[users, items].clamp(min=0.05, max=0.95)
    
    scores = torch.abs(ratings - f_pred).detach().cpu().numpy()
    weights = ((1.0 - p_hat) / p_hat).detach().cpu().numpy()
    
    q_hat = weighted_quantile(scores, weights, q=(1 - alpha))
    
    diagnostics = {
        'q_hat': float(q_hat),
        'calibration_size': len(scores),
        'scores_mean': float(np.mean(scores)),
        'scores_median': float(np.median(scores)),
        'scores_std': float(np.std(scores)),
        'scores_min': float(np.min(scores)),
        'scores_max': float(np.max(scores)),
        'scores_percentiles': {
            '25%': float(np.percentile(scores, 25)),
            '50%': float(np.percentile(scores, 50)),
            '75%': float(np.percentile(scores, 75)),
            '90%': float(np.percentile(scores, 90)),
            '95%': float(np.percentile(scores, 95)),
        }
    }
    
    if verbose:
        print(f"  Calibration diagnostics (alpha={alpha:.2f}):")
        print(f"    q_hat = {q_hat:.6f}")
        print(f"    Calibration size: {diagnostics['calibration_size']}")
        print(f"    Residual |y-f| stats: mean={diagnostics['scores_mean']:.6f}, median={diagnostics['scores_median']:.6f}")
        print(f"    Percentiles: 50%={diagnostics['scores_percentiles']['50%']:.6f}, 90%={diagnostics['scores_percentiles']['90%']:.6f}, 95%={diagnostics['scores_percentiles']['95%']:.6f}")
    
    return float(q_hat), diagnostics

def train_and_eval(train_data, val_data, test_data, ground_truth_matrix, propensity_matrix, 
                  n_user, n_item, device='cuda',
                  base_model_args: dict = {'hidden_dim': 64, 'learning_rate': 0.01, 'weight_decay': 0.1}, 
                  imputation_model_args: dict = {'hidden_dim': 10, 'learning_rate': 0.1, 'weight_decay': 0.1}, 
                  training_args: dict = {'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]},
                  alpha: float = 0.5,
                  split_ratio: float = 0.8,
                  use_conformal: bool = True,
                  conformal_strategy: str = 'correction'):
    
    if use_conformal:
        train_obs, cal_obs = split_observed(train_data, split_ratio=split_ratio)
        
        f_model = train_ips_reference_model(train_obs, propensity_matrix, device, n_user, n_item,
                                           base_model_args, training_args)
        
        p_o = train_data._nnz() / (train_data.shape[0] * train_data.shape[1])
        p_o = max(min(p_o, 1 - 1e-6), 1e-6)
        q_hat, diagnostics = compute_conformal_threshold(cal_obs, f_model, propensity_matrix, alpha, p_o, device, verbose=False)
        q_hat_tensor = torch.tensor(q_hat, device=device)
    else:
        f_model = None
        q_hat_tensor = None
    
    r_matrix = train_data.to_dense()
    observed_matrix = torch.sparse_coo_tensor(train_data._indices(), 
                                             torch.ones(train_data._values().size()).to(device), 
                                             r_matrix.size()).to_dense()
    
    imputation_model = MLP_basemodel(n_user, n_item, 
                                    hidden_dim=imputation_model_args['hidden_dim']).to(device)
    imputation_optimizer = torch.optim.Adam(imputation_model.parameters(), 
                                           lr=imputation_model_args['learning_rate'], 
                                           weight_decay=imputation_model_args['weight_decay'])
    
    base_model = MLP_basemodel(n_user, n_item, 
                              hidden_dim=base_model_args['hidden_dim']).to(device)
    base_optimizer = torch.optim.Adam(base_model.parameters(), 
                                     lr=base_model_args['learning_rate'], 
                                     weight_decay=base_model_args['weight_decay'])
    
    none_criterion = nn.MSELoss(reduction='none')
    mse_none = nn.MSELoss(reduction='none')
    
    early_stopping = SimpleEarlyStopping(patience=training_args['patience'], max_epochs=training_args['epochs'])
    
    batch_size = training_args['block_batch'][0]
    for epo in range(early_stopping.max_epochs):
        perm = torch.randperm(n_user, device=device)
        for batch_start in range(0, n_user, batch_size):
            users = perm[batch_start:min(batch_start+batch_size, n_user)]
            items = torch.randint(0, n_item, (len(users) * 10,), device=device) % n_item
            
            users_all = users.repeat_interleave(10)
            items_all = items[:len(users_all)]
            
            sub_r = r_matrix[users_all, items_all]
            sub_observed = observed_matrix[users_all, items_all]
            sub_prop = propensity_matrix[users_all, items_all].clamp(min=0.05, max=0.95)
            
            imputation_model.train()
            base_model.eval()
            
            with torch.no_grad():
                pred_fixed = base_model(users_all, items_all)
                inv_prop_fixed = 1/2 * (1.0 / sub_prop) + 1/2
                e_loss = mse_none(pred_fixed, sub_r)
            
            imp_out = imputation_model(users_all, items_all)
            e_hat_loss = mse_none(imp_out, pred_fixed)
            sq_diff = (e_loss - e_hat_loss) ** 2
            loss_imp = torch.sum(sq_diff * inv_prop_fixed * sub_observed) / (torch.sum(sub_observed) + 1e-9)
            
            imputation_optimizer.zero_grad()
            loss_imp.backward()
            imputation_optimizer.step()
            
            base_model.train()
            imputation_model.eval()
            
            inv_prop = 1/2 * (1.0 / sub_prop) + 1/2
            pred_all = base_model(users_all, items_all)
            
            with torch.no_grad():
                imp_all = imputation_model(users_all, items_all)
                if use_conformal:
                    f_pred = f_model(users_all, items_all)
            
            if use_conformal:
                if conformal_strategy == 'correction':
                    out_of_bound = (1.0 - sub_observed) * (torch.abs(imp_all - f_pred) > q_hat_tensor).float()
                    imp_corrected = imp_all.clone()
                    imp_corrected = torch.where(out_of_bound.bool(), f_pred, imp_all)
                    mask_oi = torch.ones_like(sub_observed)
                    imp_final = imp_corrected
                elif conformal_strategy == 'boundary':
                    lower_bound = f_pred - q_hat_tensor
                    upper_bound = f_pred + q_hat_tensor
                    
                    out_of_upper = (1.0 - sub_observed) * (imp_all > upper_bound).float()
                    out_of_lower = (1.0 - sub_observed) * (imp_all < lower_bound).float()
                    
                    imp_corrected = imp_all.clone()
                    imp_corrected = torch.where(out_of_upper.bool(), upper_bound, imp_corrected)
                    imp_corrected = torch.where(out_of_lower.bool(), lower_bound, imp_corrected)
                    
                    mask_oi = torch.ones_like(sub_observed)
                    imp_final = imp_corrected
                else:
                    mask_trusted = (1.0 - sub_observed) * (torch.abs(imp_all - f_pred) <= q_hat_tensor).float()
                    mask_oi = sub_observed + mask_trusted
                    imp_final = imp_all
            else:
                mask_oi = torch.ones_like(sub_observed)
                imp_final = imp_all
            
            e_true = mse_none(pred_all, sub_r)
            ips_loss = torch.sum(e_true * inv_prop * sub_observed) / (torch.sum(sub_observed) + 1e-9)
            
            e_imp = mse_none(pred_all, imp_final)
            direct_loss = torch.sum(e_imp * mask_oi) / (torch.sum(mask_oi) + 1e-9)
            
            loss_base = ips_loss + direct_loss
            
            base_optimizer.zero_grad()
            loss_base.backward()
            base_optimizer.step()
        
        base_model.eval()
        with torch.no_grad():
            all_predictions = []
            all_ground_truths = []
            
            for u_start in range(0, n_user, 100):
                u_end = min(u_start + 100, n_user)
                users_batch = torch.arange(u_start, u_end, device=device)
                
                for i_start in range(0, n_item, 100):
                    i_end = min(i_start + 100, n_item)
                    items_batch = torch.arange(i_start, i_end, device=device)
                    
                    pairs = torch.cartesian_prod(users_batch, items_batch)
                    users_eval = pairs[:, 0]
                    items_eval = pairs[:, 1]
                    
                    preds = base_model(users_eval, items_eval)
                    gts = ground_truth_matrix[users_eval, items_eval]
                    
                    all_predictions.append(preds)
                    all_ground_truths.append(gts)
            
            all_predictions = torch.cat(all_predictions)
            all_ground_truths = torch.cat(all_ground_truths)
        
        mse = torch.mean((all_predictions - all_ground_truths) ** 2).item()
        rmse = float(np.sqrt(mse))
        
        if early_stopping.check(mse, epo, base_model):
            break
    
    base_model.load_state_dict(early_stopping.best_state)
    
    base_model.eval()
    imputation_model.eval()
    if use_conformal:
        f_model.eval()
    
    with torch.no_grad():
        all_predictions = []
        all_ground_truths = []
        all_imputations_original = []
        all_imputations_corrected = []
        all_is_corrected = []
        
        for u_start in range(0, n_user, 100):
            u_end = min(u_start + 100, n_user)
            users_batch = torch.arange(u_start, u_end, device=device)
            
            for i_start in range(0, n_item, 100):
                i_end = min(i_start + 100, n_item)
                items_batch = torch.arange(i_start, i_end, device=device)
                
                pairs = torch.cartesian_prod(users_batch, items_batch)
                users_eval = pairs[:, 0]
                items_eval = pairs[:, 1]
                
                preds = base_model(users_eval, items_eval)
                gts = ground_truth_matrix[users_eval, items_eval]
                imp_orig = imputation_model(users_eval, items_eval)
                
                all_predictions.append(preds)
                all_ground_truths.append(gts)
                all_imputations_original.append(imp_orig)
                
                if use_conformal and conformal_strategy in ['correction', 'boundary']:
                    f_pred = f_model(users_eval, items_eval)
                    obs_mask = observed_matrix[users_eval, items_eval]
                    
                    if conformal_strategy == 'correction':
                        out_of_bound = (1.0 - obs_mask) * (torch.abs(imp_orig - f_pred) > q_hat_tensor).float()
                        imp_corr = torch.where(out_of_bound.bool(), f_pred, imp_orig)
                        all_imputations_corrected.append(imp_corr)
                        all_is_corrected.append(out_of_bound)
                    else:
                        lower_bound = f_pred - q_hat_tensor
                        upper_bound = f_pred + q_hat_tensor
                        
                        out_of_upper = (1.0 - obs_mask) * (imp_orig > upper_bound).float()
                        out_of_lower = (1.0 - obs_mask) * (imp_orig < lower_bound).float()
                        
                        imp_corr = imp_orig.clone()
                        imp_corr = torch.where(out_of_upper.bool(), upper_bound, imp_corr)
                        imp_corr = torch.where(out_of_lower.bool(), lower_bound, imp_corr)
                        
                        all_imputations_corrected.append(imp_corr)
                        all_is_corrected.append(out_of_upper + out_of_lower)
        
        all_predictions = torch.cat(all_predictions)
        all_ground_truths = torch.cat(all_ground_truths)
        all_imputations_original = torch.cat(all_imputations_original)
        if use_conformal and conformal_strategy in ['correction', 'boundary']:
            all_imputations_corrected = torch.cat(all_imputations_corrected)
            all_is_corrected = torch.cat(all_is_corrected)
    
    final_mse = torch.mean((all_predictions - all_ground_truths) ** 2).item()
    final_rmse = float(np.sqrt(final_mse))
    final_mae = torch.mean(torch.abs(all_predictions - all_ground_truths)).item()
    
    imputation_error_original = torch.mean(torch.abs(all_imputations_original - all_ground_truths)).item()
    
    mae_original = torch.abs(all_imputations_original - all_ground_truths).cpu().numpy()
    mae_quantiles_original = {
        '10%': float(np.percentile(mae_original, 10)),
        '30%': float(np.percentile(mae_original, 30)),
        '50%': float(np.percentile(mae_original, 50)),
        '70%': float(np.percentile(mae_original, 70)),
        '90%': float(np.percentile(mae_original, 90)),
    }
    
    epsilon = 1e-6
    mape_original = torch.mean(torch.abs((all_imputations_original - all_ground_truths) / (all_ground_truths + epsilon))).item() * 100
    
    ape_original = torch.abs((all_imputations_original - all_ground_truths) / (all_ground_truths + epsilon)).cpu().numpy() * 100
    mape_quantiles_original = {
        '10%': float(np.percentile(ape_original, 10)),
        '30%': float(np.percentile(ape_original, 30)),
        '50%': float(np.percentile(ape_original, 50)),
        '70%': float(np.percentile(ape_original, 70)),
        '90%': float(np.percentile(ape_original, 90)),
    }
    
    test_results = {
        'MSE': final_mse,
        'RMSE': final_rmse,
        'MAE': final_mae,
        'Imputation_Error': imputation_error_original,
        'MAE_Quantiles_Original': mae_quantiles_original,
        'MAPE_Original': mape_original,
        'MAPE_Quantiles_Original': mape_quantiles_original
    }
    
    if use_conformal and conformal_strategy in ['correction', 'boundary']:
        imputation_error_corrected = torch.mean(torch.abs(all_imputations_corrected - all_ground_truths)).item()
        test_results['Imputation_Error_Corrected'] = imputation_error_corrected
        test_results['Imputation_Improvement'] = imputation_error_original - imputation_error_corrected
        
        mae_corrected = torch.abs(all_imputations_corrected - all_ground_truths).cpu().numpy()
        mae_quantiles_corrected = {
            '10%': float(np.percentile(mae_corrected, 10)),
            '30%': float(np.percentile(mae_corrected, 30)),
            '50%': float(np.percentile(mae_corrected, 50)),
            '70%': float(np.percentile(mae_corrected, 70)),
            '90%': float(np.percentile(mae_corrected, 90)),
        }
        test_results['MAE_Quantiles_Corrected'] = mae_quantiles_corrected
        
        mape_corrected = torch.mean(torch.abs((all_imputations_corrected - all_ground_truths) / (all_ground_truths + epsilon))).item() * 100
        ape_corrected = torch.abs((all_imputations_corrected - all_ground_truths) / (all_ground_truths + epsilon)).cpu().numpy() * 100
        mape_quantiles_corrected = {
            '10%': float(np.percentile(ape_corrected, 10)),
            '30%': float(np.percentile(ape_corrected, 30)),
            '50%': float(np.percentile(ape_corrected, 50)),
            '70%': float(np.percentile(ape_corrected, 70)),
            '90%': float(np.percentile(ape_corrected, 90)),
        }
        test_results['MAPE_Corrected'] = mape_corrected
        test_results['MAPE_Quantiles_Corrected'] = mape_quantiles_corrected
        test_results['MAPE_Improvement'] = mape_original - mape_corrected
        
        n_total_unobserved = torch.sum(1.0 - observed_matrix).item()
        n_corrected = torch.sum(all_is_corrected).item()
        correction_rate = n_corrected / n_total_unobserved * 100
        test_results['Correction_Rate'] = correction_rate
        test_results['N_Corrected'] = int(n_corrected)
        test_results['N_Unobserved'] = int(n_total_unobserved)
    
    val_results = {
        'MSE': final_mse,
        'RMSE': final_rmse
    }
    
    return val_results, test_results


if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--p_base', type=float, default=0.4, help='Base propensity value (降低以增大偏差)')
    parser.add_argument('--propensity_type', type=str, default='mnar',
                       choices=['mnar', 'ground-truth'],
                       help='Propensity generation strategy: mnar (high rating->high obs), ground-truth (prop=rating)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--matrix_file', type=str, default='data/predicted_matrix', 
                       help='Path to predicted matrix file')
    parser.add_argument('--n_repeats', type=int, default=3, help='Repeats per alpha for averaging')
    parser.add_argument('--diagnostic', action='store_true', help='Enable diagnostic mode to show q_hat details')
    parser.add_argument('--strategy', type=str, default='correction', 
                       choices=['correction', 'boundary', 'filter'],
                       help='Conformal correction strategy: correction (midpoint), boundary (endpoint), filter (exclude)')
    
    args = parser.parse_args()
    
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading predicted matrix...")
    prediction, user_num, item_num = load_predicted_matrix(args.matrix_file)
    print(f"User num: {user_num}, Item num: {item_num}")
    
    print("Generating semisynthetic data...")
    print(f"Propensity Type: {args.propensity_type}")
    ground_truth_flat, propensity_flat, observation_flat, y_flat = generate_semisynthetic_data(
        prediction, p_base=args.p_base, propensity_type=args.propensity_type)
    
    ground_truth_matrix = torch.tensor(ground_truth_flat.reshape(user_num, item_num), 
                                      dtype=torch.float32).to(device)
    propensity_matrix = torch.tensor(propensity_flat.reshape(user_num, item_num), 
                                    dtype=torch.float32).to(device)
    
    obs_indices = np.where(observation_flat == 1)[0]
    obs_users = obs_indices // item_num
    obs_items = obs_indices % item_num
    obs_y = y_flat[obs_indices]
    
    n_obs = len(obs_indices)
    perm = np.random.permutation(n_obs)
    n_val = int(n_obs * args.val_ratio)
    n_test = int(n_obs * args.test_ratio)
    n_train = n_obs - n_val - n_test
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+n_val]
    test_idx = perm[n_train+n_val:]
    
    def build_sparse_data(idx):
        users = obs_users[idx]
        items = obs_items[idx]
        labels = obs_y[idx]
        indices = torch.tensor(np.array([users, items]), dtype=torch.long).to(device)
        values = torch.tensor(labels, dtype=torch.float32).to(device)
        return torch.sparse_coo_tensor(indices, values, torch.Size([user_num, item_num])).coalesce()
    
    train_data = build_sparse_data(train_idx)
    val_data = build_sparse_data(val_idx)
    test_data = build_sparse_data(test_idx)
    
    print(f"Observation rate: {n_obs / (user_num * item_num):.4f}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Average ground truth: {ground_truth_flat.mean():.4f}")
    print(f"Average propensity: {propensity_flat.mean():.4f}")
    
    training_args = {
        'batch_size': 1024,
        'epochs': 200,
        'patience': 30,
        'block_batch': [256, 256]
    }
    base_model_args_default = {
        'hidden_dim': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }
    imputation_model_args_default = {
        'hidden_dim': 32,
        'learning_rate': 0.005,
        'weight_decay': 1e-5
    }
    
    base_model_args = {
        'hidden_dim': base_model_args_default['hidden_dim'],
        'learning_rate': 0.001,
        'weight_decay': 0.0002
    }
    imputation_model_args = {
        'hidden_dim': imputation_model_args_default['hidden_dim'],
        'learning_rate': 0.02,
        'weight_decay': 0.002
    }
    split_ratio = 0.75
    
    baseline_mse_list = []
    baseline_mae_quantiles_list = []
    for rep in range(args.n_repeats):
        setup_seed(args.seed + rep)
        _, baseline_results = train_and_eval(
            train_data, val_data, test_data, 
            ground_truth_matrix, propensity_matrix,
            user_num, item_num,
            device,
            base_model_args=base_model_args,
            imputation_model_args=imputation_model_args,
            training_args=training_args,
            alpha=0.0,
            split_ratio=split_ratio,
            use_conformal=False
        )
        baseline_mse_list.append(baseline_results['MSE'])
        baseline_mae_quantiles_list.append(baseline_results['MAPE_Quantiles_Original'])
    
    baseline_mse = float(np.mean(baseline_mse_list))
    baseline_rmse = float(np.sqrt(baseline_mse))
    baseline_mape_quantiles = {
        '10%': float(np.mean([q['10%'] for q in baseline_mae_quantiles_list])),
        '30%': float(np.mean([q['30%'] for q in baseline_mae_quantiles_list])),
        '50%': float(np.mean([q['50%'] for q in baseline_mae_quantiles_list])),
        '70%': float(np.mean([q['70%'] for q in baseline_mae_quantiles_list])),
        '90%': float(np.mean([q['90%'] for q in baseline_mae_quantiles_list])),
    }
    
    alpha = 0.2
    mse_list = []
    mae_quantiles_list = []
    for rep in range(args.n_repeats):
        setup_seed(args.seed + rep)
        _, test_results_conf = train_and_eval(
            train_data, val_data, test_data, 
            ground_truth_matrix, propensity_matrix,
            user_num, item_num,
            device,
            base_model_args=base_model_args,
            imputation_model_args=imputation_model_args,
            training_args=training_args,
            alpha=alpha,
            split_ratio=split_ratio,
            use_conformal=True,
            conformal_strategy=args.strategy
        )
        mse_list.append(test_results_conf['MSE'])
        mae_quantiles_list.append(test_results_conf['MAPE_Quantiles_Corrected'])
    
    conformal_mse = float(np.mean(mse_list))
    conformal_rmse = float(np.sqrt(conformal_mse))
    conformal_mape_quantiles = {
        '10%': float(np.mean([q['10%'] for q in mae_quantiles_list])),
        '30%': float(np.mean([q['30%'] for q in mae_quantiles_list])),
        '50%': float(np.mean([q['50%'] for q in mae_quantiles_list])),
        '70%': float(np.mean([q['70%'] for q in mae_quantiles_list])),
        '90%': float(np.mean([q['90%'] for q in mae_quantiles_list])),
    }
    
    print('\n' + '='*60)
    print('DR-JL Baseline:')
    print(f'  RMSE: {baseline_rmse:.6f}')
    print(f'  MAPE Quantiles: 10%={baseline_mape_quantiles["10%"]:.2f}%, 30%={baseline_mape_quantiles["30%"]:.2f}%, 50%={baseline_mape_quantiles["50%"]:.2f}%, 70%={baseline_mape_quantiles["70%"]:.2f}%, 90%={baseline_mape_quantiles["90%"]:.2f}%')
    print()
    print(f'DR-JL + Conformal (alpha={alpha}):')
    print(f'  RMSE: {conformal_rmse:.6f}')
    print(f'  MAPE Quantiles: 10%={conformal_mape_quantiles["10%"]:.2f}%, 30%={conformal_mape_quantiles["30%"]:.2f}%, 50%={conformal_mape_quantiles["50%"]:.2f}%, 70%={conformal_mape_quantiles["70%"]:.2f}%, 90%={conformal_mape_quantiles["90%"]:.2f}%')
    print('='*60)
