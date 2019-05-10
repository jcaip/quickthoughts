import os
import time 
from datetime import datetime

__base_dir = os.getenv('DIR', '/home/jcjessecai/quickthoughts')

CONFIG = {
    'base_dir': __base_dir,
    'vec_path': '{}/data/GoogleNews-vectors-negative300.bin'.format(__base_dir),
    'data_path': '{}/data/cleaned.txt'.format(__base_dir),
    'checkpoint_dir': '{}/checkpoints/{:%m-%d-%H-%M-%S}'.format(__base_dir, datetime.now()), 
    'resume': False,
    'context_size': 1,
    'batch_size': 400,
    'norm_threshold': 1.0,
    'num_epochs': 5,
    'lr': 5e-4,
    'vocab_size': 10000,
}
