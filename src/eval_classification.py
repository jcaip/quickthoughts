# Experiment scripts for binary classification benchmarks (e.g. MR, CR, MPQA, SUBJ)
import os
import numpy as np
import sys
import torch

from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from utils import safe_pack_sequence
from data.utils import prepare_sequence
from numpy.random import RandomState


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

    num = 1000
    text = text[:num]
    labels = labels[:num]
    features = []
    data = []
    for i, line in enumerate(text):
        if i % 100 == 0:
            print("{:5d}/{:5d}: {}".format(i, len(text), line))
        seq, _ = prepare_sequence(line, vocab)
        data.append(torch.LongTensor(seq))
    
    packed = safe_pack_sequence(data)
    res = encoder(packed)
    features = res.detach().numpy()
    z['text'] = text
    z['labels'] = labels

    print('Computing vectors...')
    return z, features


def eval_nested_kfold(encoder, vocab, name, loc='../data/', k=10, seed=1234):
    """
    Evaluate features with nested K-fold cross validation
    Outer loop: Held-out evaluation
    Inner loop: Hyperparameter tuning

    Datasets can be found at http://nlp.stanford.edu/~sidaw/home/projects:nbsvm
    Options for name are 'MR', 'CR', 'SUBJ' and 'MPQA'
    """
    # Load the dataset and extract features
    z, features = load_data(encoder, vocab, name, loc=loc, seed=seed)

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

        scanscores = []
        for s in scan:

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
                clf = LogisticRegression(solver='liblinear', C=s)
                clf.fit(X_innertrain, y_innertrain)
                acc = clf.score(X_innertest, y_innertest)
                innerscores.append(acc)
                print("C: {:3d} Acc:{:.3f}".format(s, acc))

            # Append mean score
            scanscores.append(np.mean(innerscores))

        # Get the index of the best score
        best_score = np.max(scanscores)
        s_ind = np.argmax(scanscores)
        s = scan[s_ind]
        print("Found Best --  C:{:3d} Acc:{:.3f}".format(s, best_score))
 
        # Train classifier
        clf = LogisticRegression(solver='liblinear', C=s)
        clf.fit(X_train, y_train)

        # Evaluate
        acc = clf.score(X_test, y_test)
        scores.append(acc)

    print(scores)
    return scores

