import logging
import itertools
import time
import operator
import functools
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import DataLoader
from data.bookcorpus import BookCorpus
from qt_model import QuickThoughts
from utils import checkpoint_training, restore_training, safe_pack_sequence, VisdomLinePlotter
from config import CONFIG
from pprint import pformat, pprint
from tqdm import tqdm
import gensim.downloader as api
import os
import json

_LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    plotter = VisdomLinePlotter()
    #setting up training
    os.mkdir(CONFIG['checkpoint_dir'])
    config_filepath = "{}/{}".format(CONFIG['checkpoint_dir'], 'config.json')
    with open(config_filepath, 'w') as fp:
        _LOGGER.info(pformat(CONFIG))
        json.dump(CONFIG, fp)
    _LOGGER.info("Wrote config to file: {}".format(config_filepath))

    WV_MODEL = api.load(CONFIG['embedding'])
    # create dataset
    bookcorpus = BookCorpus(CONFIG['data_path'], WV_MODEL.vocab)
    train_iter = DataLoader(bookcorpus,
                            batch_size=CONFIG['batch_size'],
                            num_workers=1,
                            drop_last=True,
                            collate_fn=safe_pack_sequence)
    # model and loss function
    qt = QuickThoughts(WV_MODEL, CONFIG['hidden_size']).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=CONFIG['lr'])
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    last_train_idx = restore_training(CONFIG['checkpoint_dir'], qt, optimizer) if CONFIG['resume'] else -1
    #start training
    failed_or_skipped_batches = 0
    start = time.time()
    # qt.train()

    positive_block_size = 5
    #number of heads in the datset
    num_blocks = CONFIG['batch_size'] // positive_block_size
    block_offset = len(bookcorpus) // num_blocks
    heads = list(range(0, len(bookcorpus), block_offset))
    _LOGGER.info("Heads: {}".format(heads))


    def get_batch(heads, sample_size):
        examples = []
        for head in heads:
            for i in range(head, head+sample_size):
                examples.append(bookcorpus[i])
        return safe_pack_sequence(examples)

    #do this 
    for i in tqdm(range(0, block_offset)):
        
        data = get_batch(heads, positive_block_size)
        if not data:
            continue

        optimizer.zero_grad()
        data = data.cuda()
        log_scores = qt(data)
        targets = qt.generate_block_targets(positive_block_size, num_blocks)
        print(targets)

        #compute loss
        loss = kl_loss(log_scores.type(torch.cuda.DoubleTensor), targets.type(torch.cuda.DoubleTensor))
        loss.backward()
        #grad clipping
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), CONFIG['norm_threshold'])
        optimizer.step()

        if i % 10 == 0:
            _LOGGER.info("batch: {:6d} | loss: {:.4f} | failed/skipped: {:3d}".format(i, loss, failed_or_skipped_batches))

        if i % 100 == 0:
            plotter.plot('loss', 'train', 'Run: {} Loss'.format(CONFIG['checkpoint_dir'].split('/')[-1]), i, loss.item())

        if i % 10000 == 0: 
            checkpoint_training(CONFIG['checkpoint_dir'], i, qt, optimizer)

    checkpoint_training(CONFIG['checkpoint_dir'], -1, qt, optimizer, filename="FINAL_MODEL")
    _LOGGER.info("Finished Training | Total Time: {:.1f}".format(time.time()-start))


