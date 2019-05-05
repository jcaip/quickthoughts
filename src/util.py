"""
collection of helper utils
"""
from gensim.utils import tokenize
import torch
import sys
import operator
import base64
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)


def load_data(labeled_files, unlabeled_files):
    """load_data

    :param labeled_file:
    :param unlabeled_filename:
    :param num_unfrozen:
    """
    # loading nps comment dataframe
    df_labeled = pd.concat(map(pd.read_excel, labeled_files))
    df_unlabeled = pd.concat(map(pd.read_excel, unlabeled_files))

    # we shouldn't need to do this but we do :(
    df_labeled = df_labeled[df_labeled['nps_comments'].notnull()]
    df_unlabeled = df_unlabeled[df_unlabeled['nps_comments'].notnull()]

    # filter out the labeled comments
    df_unlabeled = df_unlabeled[df_unlabeled.id.apply(lambda x: x not in df_labeled.id)]

    _LOGGER.info("loaded {} labeled comments | {} unlabeled comments".format(len(df_labeled),
                                                                             len(df_unlabeled)))
    return (df_labeled, df_unlabeled)


# here tokenizing is simple, but we get rid of very common words
def tokenize_vocab(text, vocab):
    """tokenize_vocab

    :param text:
    :param vocab:
    """
    return list(filter(lambda w: w in vocab, tokenize(text.lower())))

# map to y
def categorize(label):
    return 1 if label == 'y' else 0

# formats sequence of words into a tensor
def prepare_sequence(seq, vocab):
    return torch.LongTensor([vocab[w].index for w in seq])
