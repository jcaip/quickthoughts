import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
from bookcorpus import BookCorpus
from qt_model import QuickThoughts
from pprint import pprint
from torch.utils.data.dataloader import DataLoader
from util import _LOGGER, base_dir
from util import VisdomLinePlotter

resume = False
context_size = 1
batch_size = 400
norm_threshold = 1.0 
num_epochs = 5
lr = 5e-4
data_path = "{}/cleaned.txt".format(base_dir)
checkpoint_dir = '{}/checkpoints'.format(base_dir)

#TODO: finish this metric later
def eval_batch_accuracy(scores, target):
    scores.max(1)

def safe_pack_sequence(x):
    try:
        return pack_sequence(x, enforce_sorted=False)
    except Exception as e:
        _LOGGER.exception(e)

def get_param_info(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            _LOGGER.debug("name: {} size: {}".format(name, param.data.shape))


bookcorpus = BookCorpus(data_path)
train_iter = DataLoader(bookcorpus,
                        batch_size=batch_size,
                        num_workers=10,
                        collate_fn=safe_pack_sequence)

#define our model, optimizer, and loss function
qt = QuickThoughts().cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=lr)
loss_function = nn.KLDivLoss(reduction='batchmean')
get_param_info(qt)

if resume:
    checkpoint = torch.load("{}/checkpoint_latest.pth".format(base_dir))
    qt.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_train_idx  = checkpoint['batch']
    _LOGGER.info("Resuming training from index: {}".format(last_train_idx))

else:
    _LOGGER.info("Starting training")
    last_train_idx = -1


#initializing training stuff
plotter = VisdomLinePlotter()
failed_batches = 0
running_losses = []

for i, data in enumerate(train_iter):
    #resume
    if i < last_train_idx:
        continue

    # this is to handle degnenerate sequences
    if not data:
        failed_batches +=1
        continue

    qt.zero_grad()
    data = data.cuda()
    #this gives the log softmax of the scores
    scores = qt(data)

    num_samples = min(batch_size, len(scores))
    # generate targets softmax
    targets = torch.zeros(num_samples, num_samples)
    for offset in [-1, 1]:
        targets += torch.diag(torch.ones(num_samples-abs(offset)), diagonal=offset)
    targets /= targets.sum(1, keepdim=True)
    targets = targets.cuda()

    loss = loss_function(scores, targets)
    loss.backward()
    #grad clipping
    nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), norm_threshold)
    optimizer.step()

    if i % 10 == 0:
        _LOGGER.info("batch: {:10d} | loss: {:.5f} | failed: {:4d}".format(i, loss, failed_batches))
        running_losses.append(loss.item())
        if len(running_losses) >  10:
            running_losses.pop(0)

    if i % 100 == 0:
        plotter.plot('loss', 'train', 'Loss', i, sum(running_losses) / len(running_losses))

    if i % 1000 == 0: 
        checkpoint_dict = {
            'batch': i,
            'state_dict': qt.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        savepath = "{}/checkpoint_latest.pth".format(checkpoint_dir)
        _LOGGER.info("Saving file at location : {}".format(savepath))
        torch.save(checkpoint_dict, savepath)
            
savepath = "{}/FINAL_MODEL.pth".format(checkpoint_dir)
_LOGGER.info("Saving file at location : {}".format(savepath))
torch.save(qt.state_dict(), savepath)
_LOGGER.info("Finished Training!!!")

