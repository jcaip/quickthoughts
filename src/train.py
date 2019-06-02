import logging
import itertools
import time
import operator
import functools
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_sequence, pad_sequence
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
from eval import test_performance

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
    qt = QuickThoughts(WV_MODEL, CONFIG['hidden_size'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=CONFIG['lr'])
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    last_train_idx = restore_training(CONFIG['checkpoint_dir'], qt, optimizer) if CONFIG['resume'] else -1
    #start training
    failed_or_skipped_batches = 0
    start = time.time()
    qt = qt.cuda()
    qt.train()

    # #number of heads in the datset

    # positive_block_size = 5
    # num_blocks = CONFIG['batch_size'] // positive_block_size
    # block_offset = len(bookcorpus) // num_blocks
    # heads = list(range(0, len(bookcorpus), block_offset))
    #_LOGGER.info("Heads: {}".format(heads))
    # def get_batch(heads, sample_size):
        # examples = []
        # for head in heads:
            # for i in range(head, head+sample_size):
                # examples.append(bookcorpus[i])
        # return safe_pack_sequence(examples)
    # for i in tqdm(range(0, block_offset, positive_block_size)):
        # data = get_batch(heads, positive_block_size)

    block_size=5

    for j in range(CONFIG['num_epochs']):
        temp = tqdm(train_iter)
        blocked_enc_f, blocked_enc_g= [], []
        for i, data in enumerate(temp):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            data = data.cuda()

            enc_f, enc_g = qt(data)
            blocked_enc_f.append(enc_f)
            blocked_enc_g.append(enc_g)

            #compute loss
            if len(blocked_enc_f) % block_size == 0 :
                mean_enc_f = torch.stack(blocked_enc_f).mean(dim=0)
                mean_enc_g = torch.stack(blocked_enc_g).mean(dim=0)
                #training
                scores = torch.matmul(mean_enc_f, mean_enc_g.t())
                # zero out when it's the same sentence
                mask = torch.eye(len(scores)).cuda().byte()
                scores.masked_fill_(mask, 0)    

                #return log scores and target
                log_softmax = nn.LogSoftmax(dim=1)
                block_log_scores = log_softmax(scores)
                targets = qt.generate_targets(CONFIG['batch_size'])
                loss = kl_loss(block_log_scores, targets)
                loss.backward()

                #grad clipping
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), CONFIG['norm_threshold'])
                optimizer.step()
                temp.set_description("| loss: {:.4f} | failed/skipped: {:3d}".format(loss, failed_or_skipped_batches))
                plotter.plot('loss', 'train', 'Run: {} Loss'.format(CONFIG['checkpoint_dir'].split('/')[-1]), i, loss.item())
                blocked_enc_f, blocked_enc_g= [], []

            if i % 10000 == 0: 
                checkpoint_training(CONFIG['checkpoint_dir'], i, qt, optimizer)
                qt.eval()
                acc = test_performance(qt, WV_MODEL.vocab, 'MR', '../data/rt-polaritydata')
                plotter.plot('acc', 'test', 'Run: {} Acc'.format(CONFIG['checkpoint_dir'].split('/')[-1]), i, acc)
                qt.train()

    checkpoint_training(CONFIG['checkpoint_dir'], -1, qt, optimizer, filename="FINAL_MODEL")
    _LOGGER.info("Finished Training | Total Time: {:.1f}".format(time.time()-start))
