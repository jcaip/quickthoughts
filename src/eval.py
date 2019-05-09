import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import DataLoader
from data.bookcorpus import BookCorpus
from qt_model import QuickThoughts
from utils import checkpoint_training, restore_training, safe_pack_sequence
from config import CONFIG
from pprint import pformat
import os
import json
import eval_classification

_LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':

    # init wordvec model
    WV_MODEL = KeyedVectors.load_word2vec_format(CONFIG['vec_path'], binary=True, limit=CONFIG['vocab_size'])

    start = time.time()
    # model and loss function
    qt = QuickThoughts(WV_MODEL, cuda=False)
    trained_params = torch.load("{}/data/FINAL_MODEL.pth".format(CONFIG['base_dir']))
    qt.load_state_dict(trained_params)
    _LOGGER.info("Restored successfully")
    qt.eval()

    eval_classification.eval_nested_kfold(qt, WV_MODEL.vocab, 'MR')


    end = time.time()
    _LOGGER.info("Finished Eval | | Total Time: {}".format(end-start))


