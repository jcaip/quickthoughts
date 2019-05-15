import logging
import os
import json
import numpy as np
import sys
import time
import torch
import torch.nn as nn
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
import operator
import torch.optim as optim
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import DataLoader
from data.bookcorpus import BookCorpus
from qt_model import QuickThoughts
from utils import checkpoint_training, restore_training, safe_pack_sequence
from config import CONFIG
from pprint import pformat
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, shuffle
from data.utils import prepare_sequence
from numpy.random import RandomState

pool = Pool(6)

def load_data(encoder, vocab, name, loc='./data/', seed=1234):
    z = {}
    if name == 'MR':
        pos, neg = [], []
        with open(os.path.join(loc, 'rt-polarity.pos'), 'rb') as f:
            for line in f:
                pos.append(line.decode('latin-1').strip())
        with open(os.path.join(loc, 'rt-polarity.neg'), 'rb') as f:
            for line in f:
                neg.append(line.decode('latin-1').strip())

    labels = compute_labels(pos, neg)
    text, labels = shuffle_data(pos+neg, labels, seed=seed)
    test_batch_size = 1000
    size = len(text)
    feature_list = []
    for j in range(0, size, test_batch_size):
        stop_idx = min(len(text), j+1000)
        _LOGGER.info("Processing data from {} to {}".format(j, stop_idx))
        batch_text  = text[j:stop_idx]
        batch_labels = labels[j:stop_idx]
        data = [torch.LongTensor(seq) for seq, _ in map(lambda x: prepare_sequence(x, vocab), batch_text)]
            # if i % 100 == 0:
                # print("{:5d}/{:5d}: {}".format(i, len(batch_text), line))
        packed = safe_pack_sequence(data).cuda()
        res = encoder(packed).cpu().detach().numpy()
        feature_list.append(res)

    features = np.concatenate(feature_list)
    _LOGGER.info("featurs of shape: {}".format(features.shape))
    z['text'] = text
    z['labels'] = labels

    return z, features


def eval_nested_kfold(encoder, vocab, name, loc='../data/', k=10, seed=1234):
    """
    Evaluate features with nested K-fold cross validation
    Outer loop: Held-out evaluation
    Inner loop: Hyperparameter tuning

    Options for name are 'MR', 'CR', 'SUBJ' and 'MPQA'
    """
    # Load the dataset and extract features
    z, features = load_data(encoder, vocab, name, loc=loc, seed=seed)

    _LOGGER.info("Fitting logistic layers")

    scan = [2**t for t in range(0,9,1)]
    npts = len(z['text'])
    kf = KFold(n_splits=k, random_state=seed)
    scores = []
    for train, test in kf.split(z['labels']):
        # Split data
        X_train = features[train]
        y_train = z['labels'][train]
        X_test = features[test]
        y_test = z['labels'][test]

        Xraw = [z['text'][i] for i in train]
        Xraw_test = [z['text'][i] for i in test]

        def fit_inner_kfold(s):
            # Inner KFold
            innerkf = KFold(n_splits=k, random_state=seed+1)
            innerscores = []
            for innertrain, innertest in innerkf.split(X_train):

                # Split data
                X_innertrain = X_train[innertrain]
                y_innertrain = y_train[innertrain]
                X_innertest = X_train[innertest]
                y_innertest = y_train[innertest]

                Xraw_innertrain = [Xraw[i] for i in innertrain]
                Xraw_innertest = [Xraw[i] for i in innertest]

                # Train classifier
                clf = LogisticRegression(solver='lbfgs', C=s)
                clf.fit(X_innertrain, y_innertrain)
                acc = clf.score(X_innertest, y_innertest)
                innerscores.append(acc)
                # print("C: {:3d} Acc:{:.3f}".format(s, acc))

            # Append mean score
            return (s, np.mean(innerscores))

        start = time.time()
        scanscores = pool.map(fit_inner_kfold, scan)

        # Get the index of the best score
        s, best_score = max(scanscores, key=operator.itemgetter(1))
 
        # Train classifier
        clf = LogisticRegression(solver='lbfgs', C=s)
        clf.fit(X_train, y_train)

        # Evaluate
        acc = clf.score(X_test, y_test)
        scores.append(acc)

        _LOGGER.info("Found best C={:3d} with accuracy: {:.2%} in {:.2f} seconds | Test Accuracy: {:.2%}".format(s, best_score, time.time()-start, acc))

    return scores


_LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':

    start = time.time()


    qt = QuickThoughts(WV_MODEL).cuda()
    trained_params = torch.load("{}/checkpoints/05-13-18-46-07/FINAL_MODEL.pth".format(CONFIG['base_dir']))
    qt.load_state_dict(trained_params['state_dict'])
    qt.eval()

    _LOGGER.info("Restored successfully")

    scores = eval_nested_kfold(qt, WV_MODEL.vocab, 'MR')

    end = time.time()
    _LOGGER.info("Finished Evaluation of {} | Accuracy: {} | Total Time: {}".format('MR', np.mean(scores), end-start))


