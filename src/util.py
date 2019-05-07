import logging
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from visdom import Visdom
import numpy as np
import torch

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

base_dir = '/home/jcaip/workspace/quickthoughts'
vec_path = '{}/GoogleNews-vectors-negative300.bin'.format(base_dir)
_WV_MODEL = KeyedVectors.load_word2vec_format(vec_path, binary=True, limit=10000)

def prepare_sequence(text, vocab=_WV_MODEL.vocab, max_len=50, return_tensor=True):
    pruned_sequence = zip(filter(lambda x: x in vocab, text), range(max_len))
    seq = [vocab[x].index for x, _ in pruned_sequence]
    if return_tensor:
        return torch.LongTensor(seq)
    return seq

class VisdomLinePlotter(object):

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='batch',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
