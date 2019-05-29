import logging
import torch
from visdom import Visdom
from torch.nn.utils.rnn import pack_sequence
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

def safe_pack_sequence(x):
    try:
        packed_batch = pack_sequence(x, enforce_sorted=False)
        # targets = torch.zeros(len(x), len(x))
        # for i, t1 in enumerate(x):
            # for j in range(i+1, len(x)):
                # targets[i, j] = len(np.setdiff1d(t1.numpy(),x[j].numpy()))
        # targets += targets.t()

        return packed_batch
            
    except Exception as e:
        _LOGGER.exception(e)

def log_param_info(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            _LOGGER.debug("name: {} size: {}".format(name, param.data.shape))

def checkpoint_training(checkpoint_dir, idx, model, optim, filename="checkpoint_latest"):
    """checkpoint training, saves optimizer, model, and index"""
    checkpoint_dict = {
        'batch': idx,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
    }
    savepath = "{}/{}.pth".format(checkpoint_dir, filename)
    _LOGGER.info("Saving file at location : {}".format(savepath))
    torch.save(checkpoint_dict, savepath)

def restore_training(checkpoint_dir, model, optimizer, filename="checkpoint_latest"):
    """restore training from a checkpoint dir, returns batch"""
    checkpoint = torch.load("{}/{}.pth".format(checkpoint_dir, filename))
    print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    _LOGGER.info("Resuming training from index: {}".format(checkpoint['batch']))
    return checkpoint['batch']

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

