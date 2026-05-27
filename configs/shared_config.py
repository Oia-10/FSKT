import os

# =============================================================================
CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CONFIGS_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# =============================================================================
DATASET_CONFIG = {
    'assist2009': {'num_q': 17737, 'num_c': 123},
    'assist2012': {'num_q': 53070, 'num_c': 265},
    'assist2017': {'num_q': 3162, 'num_c': 102},
    'algebra2005': {'num_q': 173113, 'num_c': 112},
    'bridge2algebra2006': {'num_q': 129263, 'num_c': 493}
}


# =============================================================================
TRAIN_CONFIG = {
    'seq_len': 200,           
    'batch_size': 128,         
    'epochs': 200,          
    'early_stop_patience': 10, 
    'seed': 1010,            
    'n_folds': 5,             
    'grad_clip': 10,           
    'lr_decay_step': 1000,      
    'lr_decay_rate': 0.5,      
    'weight_decay': 0,        
}

# =============================================================================
MODEL_CONFIG = {
    'embed_dim': 128,        
}


# =============================================================================
def get_data_paths(dataset):
    dataset_dir = os.path.join(DATA_DIR, dataset)
    return {
        'data': os.path.join(dataset_dir, f'{dataset}_pro.csv'),
        'test_split': os.path.join(dataset_dir, f'test_{dataset}_split.pkl'),
        'train_valid_split': os.path.join(dataset_dir, '{cv}_train_valid_{dataset}_split.pkl'.replace('{dataset}', dataset)),
        'q_matrix': os.path.join(dataset_dir, f'{dataset}_q_matrix.npy'),
    }


def get_dataset_info(dataset):
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unknown: {dataset}, Optional: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset]
