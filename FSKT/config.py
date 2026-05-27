import os
import sys
import json
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import DATASET_CONFIG, TRAIN_CONFIG, MODEL_CONFIG, get_data_paths, PROJECT_DIR

# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'result')
LOCAL_CONFIGS_DIR = os.path.join(BASE_DIR, 'configs')


def get_num_workers():
    return 0 if sys.platform == 'win32' else 4

# =============================================================================
def set_opt():
    parser = ArgumentParser(description='FSKT')

    parser.add_argument('--dataset', type=str, default='assist2009',
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--length', type=int, default=TRAIN_CONFIG['seq_len'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=TRAIN_CONFIG['weight_decay'])
    parser.add_argument('--early_stop_patience', type=int, default=TRAIN_CONFIG['early_stop_patience'])
    parser.add_argument('--seed', type=int, default=TRAIN_CONFIG['seed'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--grad_clip', type=float, default=TRAIN_CONFIG['grad_clip'])
    parser.add_argument('--lr_decay_step', type=int, default=TRAIN_CONFIG['lr_decay_step'])
    parser.add_argument('--lr_decay_rate', type=float, default=TRAIN_CONFIG['lr_decay_rate'])
    parser.add_argument('--embed_dim', type=int, default=MODEL_CONFIG['embed_dim'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_levels', type=int, default=2)
    parser.add_argument('--lambda1', type=float, default=0.01)

    opt = parser.parse_args()

    opt.num_q = DATASET_CONFIG[opt.dataset]['num_q']
    opt.num_c = DATASET_CONFIG[opt.dataset]['num_c']
    opt.paths = get_data_paths(opt.dataset)

    return opt
