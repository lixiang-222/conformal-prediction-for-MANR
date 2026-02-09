import argparse

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='learning framework for RS')
    parser.add_argument('--dataset', type=str, default='coat', help='Choose from {yahooR3, coat, kuaiRec}')
    parser.add_argument('--base_model_args', type=dict, default={'emb_dim': 10, 'learning_rate': 0.01, 'imputaion_lambda': 0.01, 'weight_decay': 1}, 
                help='base model arguments.')
    parser.add_argument('--imputation_model_args', type=dict, default= {'learning_rate': 1e-1, 'weight_decay': 1e-4}, 
                help='imputation model arguments.')          
    parser.add_argument('--training_args', type=dict, default = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [20, 500]}, 
                help='training arguments.')
    parser.add_argument('--uniform_ratio', type=float, default=0.05, help='the ratio of uniform set in the unbiased dataset.')
    parser.add_argument('--alpha', type=float, default=0.5, help='conformal coverage level.')
    parser.add_argument(
        '--conformal_policy',
        type=str,
        default='fpred',
        choices=['hard_reject', 'clip', 'fpred'],
        help='how to use conformal threshold for O=0 samples.'
    )
    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    return parser.parse_args()
