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

_LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':

    #setting up training
    os.mkdir(CONFIG['checkpoint_dir'])
    config_filepath = "{}/{}".format(CONFIG['checkpoint_dir'], 'config.json')
    with open(config_filepath, 'w') as fp:
        _LOGGER.info(pformat(CONFIG))
        json.dump(CONFIG, fp)
        _LOGGER.info("Wrote config to file: {}".format(config_filepath))


    # init wordvec model
    WV_MODEL = KeyedVectors.load_word2vec_format(CONFIG['vec_path'], binary=True, limit=CONFIG['vocab_size'])

    # create dataset
    bookcorpus = BookCorpus(CONFIG['data_path'], WV_MODEL.vocab)
    train_iter = DataLoader(bookcorpus,
                            batch_size=CONFIG['batch_size'],
                            num_workers=1,
                            drop_last=True,
                            collate_fn=safe_pack_sequence)

    # model and loss function
    qt = QuickThoughts(WV_MODEL).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=CONFIG['lr'])
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    last_train_idx = restore_training(CONFIG['checkpoint_dir'], qt, optimizer) if CONFIG['resume'] else -1

    #start training
    failed_or_skipped_batches = 0
    start = time.time()
    qt.train()

    for i, data in enumerate(train_iter):
        # this handles resuming / when we have a bad sample (0 -len sequence)
        if i < last_train_idx or not data:
            failed_or_skipped_batches += 1
            continue

        optimizer.zero_grad()
        data = data.cuda()

        log_scores = qt(data)
        targets = qt.generate_targets(CONFIG['batch_size'])

        #compute loss
        loss = kl_loss(log_scores, targets)
        loss.backward()
        #grad clipping
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), CONFIG['norm_threshold'])
        optimizer.step()

        if i % 10 == 0:
            _LOGGER.info("batch: {:6d} | loss: {:.4f} | failed/skipped: {:3d}".format(i, loss, failed_or_skipped_batches))

        if i % 10000 == 0: 
            checkpoint_training(CONFIG['checkpoint_dir'], i, qt, optimizer)

    checkpoint_training(CONFIG['checkpoint_dir'], -1, qt, optimizer, filename="FINAL_MODEL")
    end = time.time()
    _LOGGER.info("Finished Training | Total Time: {}".format(end-start))


