"""
Evaluation script for a variety of datasets.
This is a slightly modified version of the original
skip-thoughts eval code, found here:

https://github.com/ryankiros/skip-thoughts/blob/master/eval_classification.py
"""
import logging
import os
import json
import sys
import time
import operator
import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool as Pool
import torch.optim as optim
from gensim.models import KeyedVectors
import gensim.downloader as api
from data.bookcorpus import BookCorpus
from qt_model import QuickThoughts
from utils import checkpoint_training, restore_training, safe_pack_sequence
from config import CONFIG
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from data.utils import prepare_sequence
from numpy.random import RandomState

_LOGGER = logging.getLogger(__name__)
pool = Pool(6)

def load_data(encoder, vocab, name, loc='./data/', seed=1234):
    if name == 'MR':
        with open(os.path.join(loc, 'rt-polarity.pos'), 'rb') as f:
            pos = [line.decode('latin-1').strip() for line in f]
        with open(os.path.join(loc, 'rt-polarity.neg'), 'rb') as f:
            neg = [line.decode('latin-1').strip() for line in f]

    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    text, labels = shuffle(pos+neg, labels, random_state=seed)
    size = len(labels)
    _LOGGER.info("Loaded dataset {} with total lines: {}".format(name, size))

    def make_batch(j):
        stop_idx = min(size, j+CONFIG['test_batch_size'])
        _LOGGER.info("Processing data from {} to {}".format(j, stop_idx))

        batch_text, batch_labels  = text[j:stop_idx], labels[j:stop_idx]
        data = [torch.LongTensor(seq) for seq, _ in map(lambda x: prepare_sequence(x, vocab), batch_text)]
        packed = safe_pack_sequence(data).cuda()

        return encoder(packed).cpu().detach().numpy()

    feature_list = [make_batch(i) for i in range(0, size, CONFIG['test_batch_size'])]
    features = np.concatenate(feature_list)
    _LOGGER.info("Test feature matrix of shape: {}".format(features.shape))

    return text, labels, features


def eval_nested_kfold(encoder, vocab, name, loc='../data/', k=10, seed=1234):
    # Load the dataset and extract features
    text, labels, features = load_data(encoder, vocab, name, loc=loc, seed=seed)
    _LOGGER.info("Fitting logistic layers")

    scan = [ 2**t for t in range(8) ]
    npts = len(text)

    def fit_clf(X_train, y_train, X_test, y_test, s):
        clf = LogisticRegression(solver='sag', C=s)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        _LOGGER.info("Fitting logistic model with s: {:.3d} and acc: {:.2%}".format(s, acc))
        return acc

    def chunk_data(train_idx, test_idx, X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        return (X_train, y_train, X_test, y_test)

    def fit_outer_kfold(train, test):

        X_train, y_train, X_test, y_test = chunk_data(train, test, features, labels)

        start = time.time()

        def fit_inner_kfold(s):
            innerkf = KFold(n_splits=k, random_state=seed+1)
            innerscores = [fit_clf(*chunk_data(*train_test_idx, X_train, y_train), s) for train_test_idx in innerkf.split(X_train)]
            return (s, np.mean(innerscores))

        scanscores = pool.map(fit_inner_kfold, scan)
        s, best_score = max(scanscores, key=operator.itemgetter(1))
        acc = fit_clf(X_train, y_train, X_test, y_test, s)
        _LOGGER.info("Found best C={:3d} with accuracy: {:.2%} in {:.2f} seconds | Test Accuracy: {:.2%}".format(s, best_score, time.time()-start, acc))

        return acc

    kf = KFold(n_splits=k, random_state=seed)
    return [fit_outer_kfold(*train_test_idx) for train_test_idx in kf.split(labels)]



if __name__ == '__main__':

    start = time.time()

    WV_MODEL = KeyedVectors.load_word2vec_format(CONFIG['vec_path'], binary=True, limit=CONFIG['vocab_size'])
    qt = QuickThoughts(WV_MODEL, hidden_size=1000).cuda()
    trained_params = torch.load("{}/data/FINAL_MODEL.pth".format(CONFIG['base_dir']))
    qt.load_state_dict(trained_params)
    qt.eval()

    _LOGGER.info("Restored successfully")
    scores = eval_nested_kfold(qt, WV_MODEL.vocab, 'MR')
    end = time.time()
    _LOGGER.info("Finished Evaluation of {} | Accuracy: {:.2%} | Total Time: {:.1f}".format('MR', np.mean(scores), end-start))


