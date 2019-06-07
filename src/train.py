import logging
import itertools
import time
import operator
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    #load in word vectors
    WV_MODEL = api.load(CONFIG['embedding'])

    # create dataset
    bookcorpus = BookCorpus(CONFIG['data_path'], WV_MODEL.vocab)
    train_iter = DataLoader(bookcorpus,
                            batch_size=CONFIG['batch_size'],
                            num_workers=1,
                            drop_last=True,
                            pin_memory=True, #send to GPU
                            collate_fn=safe_pack_sequence)

    # model, optimizer, and loss function
    qt = QuickThoughts(WV_MODEL, CONFIG['hidden_size']).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=CONFIG['lr'])
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    #start training
    qt = qt.train()
    failed_or_skipped_batches = 0
    last_train_idx = restore_training(CONFIG['checkpoint_dir'], qt, optimizer) if CONFIG['resume'] else -1
    start = time.time()
    block_size=5

    #training loop
    for j in range(CONFIG['num_epochs']):
        temp = tqdm(train_iter)

        for i, data in enumerate(temp):
            #deal with bad sequences
            if not data:
                failed_or_skipped_batches+=1
                continue

            #zero out
            try:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                data = data.cuda()

                #forward pass
                enc_f, enc_g = qt(data)


                # for k in range(0, CONFIG['batch_size'], block_size):
                    # avg = enc_g[k:k+block_size].mean(dim=0)
                    # for l in range(block_size):
                        # enc_g[k+l, :] = avg - enc_g[k+l, :] 

                # calculate scores
                scores = torch.matmul(enc_f, enc_g.t())
                # zero out when it's the same sentence
                mask = torch.eye(len(scores)).cuda().byte()
                scores.masked_fill_(mask, 0)    

                #return log scores and target
                block_log_scores = F.log_softmax(scores, dim=1)
                # targets also topelitz matrix
                targets = qt.generate_targets(CONFIG['batch_size'], offsetlist=[1])
                loss = kl_loss(block_log_scores, targets)
                loss.backward()
        
                #grad clipping
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), CONFIG['norm_threshold'])
                optimizer.step()

                temp.set_description("loss {:.4f} | failed/skipped {:3d}".format(loss, failed_or_skipped_batches))

                if i % 100 == 0:
                    plotter.plot('loss', 'train', 'Run: {} Loss'.format(CONFIG['checkpoint_dir'].split('/')[-1]), i, loss.item())

                if i % 5000 == 0: 
                    checkpoint_training(CONFIG['checkpoint_dir'], i, qt, optimizer)
                    qt.eval()
                    for dataset in ['MR', 'CR', 'MPQA', 'SUBJ']:
                        acc = test_performance(qt, WV_MODEL.vocab, dataset, '../data', seed=int(time.time()))
                        plotter.plot('acc', dataset, 'Downstream Accuracy', i, acc, xlabel='seconds')
                    qt.train()

            except Exception as e:
                _LOGGER.exception(e)
                failed_or_skipped_batches+=1
                torch.cuda.empty_cache()

    checkpoint_training(CONFIG['checkpoint_dir'], -1, qt, optimizer, filename="FINAL_MODEL")
    _LOGGER.info("Finished Training | Total Time: {:.1f}".format(time.time()-start))

