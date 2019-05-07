import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from data.bookcorpus import BookCorpus
from qt_model import QuickThoughts
from pprint import pprint
from util import prepare_sequence, VisdomLinePlotter
from torch.utils.data.dataloader import DataLoader
from util import _LOGGER

cuda = torch.device('cuda')
batch_size = 400
context_size = 1
lr = 5e-4
norm_threshold=1.0 

data_path = '/home/jcjessecai/quickthoughts/all.txt'
bc = BookCorpus(data_path)
train_iter = DataLoader(bc, batch_size=batch_size, num_workers=10, collate_fn=pad_sequence)

#define our model
qt = QuickThoughts()
qt = qt.cuda()

# some debugging
for name, param in qt.named_parameters():
    if param.requires_grad:
        _LOGGER.info(name, param.data.size())

plotter = VisdomLinePlotter()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=lr)
loss_function = nn.KLDivLoss(reduction='batchmean')

_LOGGER.info("Starting training")

def eval_batch_accuracy(scores, target):
    scores.max(1)

for i, data in enumerate(train_iter):
    try:
        qt.zero_grad()
        data = data.cuda()
        scores = qt(data)

        #does this all on the gpu so hopefully faster
        targets = torch.zeros(batch_size, batch_size)
        for offset in [-1, 1]:
            targets += torch.diag(torch.ones(batch_size-1), diagonal=offset)
        targets = targets / targets.sum(1, keepdim=True)
        targets = targets.cuda()


        loss = loss_function(scores, targets)
        
        if i % 10 == 0:
            _LOGGER.info("batch: {} loss: {}".format(i, loss))

        if i % 100 == 0:
            plotter.plot('loss', 'train', 'Loss', i, loss.item())

        if i % 1000 == 0: 
            test_sentences =  [ "What is going on?",
                               "Let's go eat.",
                               "The engine won't start.",
                               "I'm hungry now."]

            pprint(test_sentences)

            test_sentences = pad_sequence(list(map(prepare_sequence, test_sentences))).cuda()
            print(torch.exp(qt(test_sentences)))

            savepath = "../checkpoints/model-{}.pth".format(i)
            print("Saving file at location : {}".format(savepath))

            torch.save(qt.state_dict(), savepath)

        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), norm_threshold)
        optimizer.step()

    except Exception as e:
        # print("Whoops: {}".format(e))
        _LOGGER.exception(e)

