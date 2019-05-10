import logging
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
import os
import json
import numpy as np
import sys
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from data.utils import prepare_sequence
from numpy.random import RandomState


pool = Pool(6)

def shuffle_data(X, L, seed=1234):
    """
    Shuffle the data
    """
    prng = RandomState(seed)
    inds = np.arange(len(X))
    prng.shuffle(inds)
    X = [X[i] for i in inds]
    L = L[inds]
    return (X, L)    

def compute_labels(pos, neg):
    """
    Construct list of labels
    """
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1.0
    labels[len(pos):] = 0.0
    return labels

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
        print("Processing data from {} to {}".format(j, stop_idx))
        batch_text  = text[j:stop_idx]
        batch_labels = labels[j:stop_idx]
        data = [torch.LongTensor(seq) for seq, _ in map(lambda x: prepare_sequence(x, vocab), batch_text)]
            # if i % 100 == 0:
                # print("{:5d}/{:5d}: {}".format(i, len(batch_text), line))
        packed = safe_pack_sequence(data).cuda()
        res = encoder(packed).cpu().detach().numpy()
        feature_list.append(res)

    features = np.concatenate(feature_list)
    print(features.shape)
    z['text'] = text
    z['labels'] = labels

    print('Computing vectors...')
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

    print("evaluating")

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
        print("Finished C search in {:.2f} seconds".format(time.time()-start))

        # Get the index of the best score
        s, best_score = max(scanscores, key=operator.itemgetter(1))
        print("Found Best --- C:{:3d} Acc:{:.3f}".format(s, best_score))
 
        # Train classifier
        clf = LogisticRegression(solver='lbfgs', C=s)
        clf.fit(X_train, y_train)

        # Evaluate
        acc = clf.score(X_test, y_test)
        print("test acc: {:.3f}".format(acc))
        scores.append(acc)

    print(scores)
    print(np.mean(scores))
    return scores


_LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':

    # init wordvec model
    WV_MODEL = KeyedVectors.load_word2vec_format(CONFIG['vec_path'], binary=True, limit=CONFIG['vocab_size'])

    start = time.time()
    # model and loss function
    qt = QuickThoughts(WV_MODEL).cuda()
    trained_params = torch.load("{}/data/FINAL_MODEL.pth".format(CONFIG['base_dir']))
    qt.load_state_dict(trained_params)
    _LOGGER.info("Restored successfully")
    qt.eval()
    qt.training = False

    eval_nested_kfold(qt, WV_MODEL.vocab, 'MR')


    end = time.time()
    _LOGGER.info("Finished Eval | | Total Time: {}".format(end-start))


