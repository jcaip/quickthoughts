import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from bookcorpus import BookCorpus
from qt_model import QuickThoughts
from pprint import pprint
from util import VisdomLinePlotter
from torch.utils.data.dataloader import DataLoader
from util import _LOGGER, base_dir

resume = False
context_size = 1
batch_size = 400
norm_threshold = 1.0 
num_epochs = 5
lr = 5e-4

data_path = "{}/old_cleaned.txt".format(base_dir)
checkpoint_dir = '{}/checkpoints'.format(base_dir)

bookcorpus = BookCorpus(data_path)
train_iter = DataLoader(bookcorpus,
                        batch_size=batch_size,
                        num_workers=10,
                        collate_fn=lambda x: pack_sequence(x, enforce_sorted=False))

#define our model
qt = QuickThoughts().cuda()

# some debugging info
for name, param in qt.named_parameters():
    if param.requires_grad:
        _LOGGER.info("name: {} size: {}".format(name, param.data.shape))

#initializing
plotter = VisdomLinePlotter()

#optimizer and loss function
optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=lr)

loss_function = nn.KLDivLoss(reduction='batchmean')

if resume:
    _LOGGER.info("Resuming training!")
    checkpoint = torch.load("{}/checkpoint_latest.pth".format(base_dir))
    qt.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

else:
    _LOGGER.info("Starting training")
    start_epoch = 0

#TODO: finish this metric later
def eval_batch_accuracy(scores, target):
    scores.max(1)

def show_test_data_similarity(qt):
    # test_sentences =  [ "What is going on?",
                        # "Let's go eat.",
                        # "The engine won't start.",
                        # "I'm hungry now."]

    # pprint(test_sentences)
    # test_sentences = pack_sequence(list(map(prepare_sequence, test_sentences)), enforce_sorted=False).cuda()
    # print(torch.exp(qt(test_sentences)))
    pass


for j in range(start_epoch, num_epochs):

    running_losses = []

    for i, data in enumerate(train_iter):
        try:
            qt.zero_grad()

            data = data.cuda()
            #this gives the log softmax of the scores
            scores = qt(data)

            # generate targets softmax
            targets = torch.zeros(batch_size, batch_size)
            for offset in [-1, 1]:
                targets += torch.diag(torch.ones(batch_size-abs(offset)), diagonal=offset)
            targets /= targets.sum(1, keepdim=True)
            targets = targets.cuda()

            loss = loss_function(scores, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), norm_threshold)
            optimizer.step()

            if i % 10 == 0:
                _LOGGER.info("epoch: {} batch: {:10d} loss: {:.5f}".format(j, i, loss))
                running_losses.append(loss.item())
                if len(running_losses) >  10:
                    running_losses.pop(0)

            if i % 100 == 0:
                plotter.plot('loss', 'train', 'Loss', i, sum(running_losses) / len(running_losses))

            if i % 1000 == 0: 
                show_test_data_similarity(qt)
                checkpoint_dict = {
                    'epoch': j+1,
                    'state_dict': qt.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                savepath = "{}/checkpoint_latest.pth".format(checkpoint_dir, i)
                _LOGGER.info("Saving file at location : {}".format(savepath))
                torch.save(checkpoint_dict, savepath)
            
        except Exception as e:

            _LOGGER.exception(e)
            print("---------\n\n")



