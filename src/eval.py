"""
Evaluation script for a variety of datasets.
This is a slightly modified version of the original
skip-thoughts eval code, found here:

https://github.com/ryankiros/skip-thoughts/blob/master/eval_classification.py
"""
import json
import logging
import operator
import os
import time
import torch
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from pathos.multiprocessing import ProcessingPool as Pool
from data.utils import prepare_sequence
from data.bookcorpus import BookCorpus
from utils import checkpoint_training, restore_training, safe_pack_sequence
from qt_model import QuickThoughts

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

def load_data(encoder, vocab, name, loc, seed=1234, test_batch_size=1000):
    """load in a binary classification dataste for evaluation"""

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
        """Processes one test batch of the test datset"""
        stop_idx = min(size, j+test_batch_size)
        _LOGGER.info("Processing data from {:5d} to {:5d}".format(j, stop_idx))
        batch_text, batch_labels  = text[j:stop_idx], labels[j:stop_idx]
        data = [torch.LongTensor(seq) for seq in map(lambda x: prepare_sequence(x, vocab), batch_text)]
        packed = safe_pack_sequence(data).cuda()

        return encoder(packed).cpu().detach().numpy()

    feature_list = [make_batch(i) for i in range(0, size, test_batch_size]
    features = np.concatenate(feature_list)
    _LOGGER.info("Test feature matrix of shape: {}".format(features.shape))

    return text, labels, features

def fit_clf(X_train, y_train, X_test, y_test, s):
    """Fits a single classifier and returns test accuracy"""
    clf = LogisticRegression(solver='sag', C=s)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    #_LOGGER.info("Fitting logistic model with s: {:3d} and acc: {:.2%}".format(s, acc))
    return acc

def chunk_data(train_idx, test_idx, X, y):
    """returns a split of the data""""
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train, X_test, y_test)

def eval_nested_kfold(encoder, vocab, name, loc='../data/rt-polaritydata', k=10, seed=1234):
    """Evaluates nested kfold to get accuracy"""

    def fit_outer_kfold(train, test):
        """fits a single outer kfold"""
        start = time.time()
        X_train, y_train, X_test, y_test = chunk_data(train, test, features, labels)

        def fit_inner_kfold(s):
            innerkf = KFold(n_splits=k, random_state=seed+1)
            innerscores = [fit_clf(*chunk_data(*train_test_idx, X_train, y_train), s) for train_test_idx in innerkf.split(X_train)]
            return (s, np.mean(innerscores))

        scanscores = [fit_inner_kfold(s) for s in scan]
        s, best_score = max(scanscores, key=operator.itemgetter(1))
        acc = fit_clf(X_train, y_train, X_test, y_test, s)
        _LOGGER.info("Found best C={:3d} with accuracy: {:.2%} in {:.2f} seconds | Test Accuracy: {:.2%}".format(s, best_score, time.time()-start, acc))
        return acc

    text, labels, features = load_data(encoder, vocab, name, loc=loc, seed=seed)
    scan = [ 2**t for t in range(1) ]
    npts = len(text)
    pool = Pool(4)
    kf = KFold(n_splits=k, random_state=seed)
    scores = pool.map(lambda x: fit_outer_kfold(*x), kf.split(labels))
    return scores

if __name__ == '__main__':
    start = time.time()
    checkpont_dir = '/home/jcaip/quickthoughts/checkpoints/'
    CONFIG = json.loads("{}/config.json".format(checkpoint_dir))

    WV_MODEL = api.load(CONFIG['embedding'])
    qt = QuickThoughts(WV_MODEL, hidden_size=CONFIG['hidden_size']).cuda()
    trained_params = torch.load("{}/FINAL_MODEL.pth".format(checkpoint_dir)
    qt.load_state_dict(trained_params['state_dict'])
    qt.eval()

    _LOGGER.info("Restored successfully from {}".format(checkpoint_dir))
    scores = eval_nested_kfold(qt, WV_MODEL.vocab, 'MR')
    _LOGGER.info("Finished Evaluation of {} | Accuracy: {:.2%} | Total Time: {:.1f}s".format('MR', np.mean(scores), time.time()-start))
