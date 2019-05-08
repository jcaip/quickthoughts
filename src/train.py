import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import DataLoader
from bookcorpus import BookCorpus
from qt_model import QuickThoughts
from config import CONFIG
from util import VisdomLinePlotter, WV_MODEL, checkpoint_training, safe_pack_sequence, restore_training

_LOGGER = logging.getLogger(__name__)

bookcorpus = BookCorpus(CONFIG['data_path'])
train_iter = DataLoader(bookcorpus,
                        batch_size=CONFIG['batch_size'],
                        num_workers=10,
                        collate_fn=safe_pack_sequence)
qt = QuickThoughts(WV_MODEL).cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=CONFIG['lr'])
loss_function = nn.KLDivLoss(reduction='batchmean')
last_train_idx = restore_training(CONFIG['checkpoint_dir']) if CONFIG['resume'] else -1
plotter = VisdomLinePlotter()
failed_or_skipped_batches, running_losses, start = 0, [], time.time()

for i, data in enumerate(train_iter):
    # this handles resuming / when we have a bad sample (0 -len sequence)
    if i < last_train_idx or not data:
        failed_or_skipped_batches += 1
        continue
    qt.zero_grad()
    data = data.cuda()
    log_scores = qt(data)
    targets = qt.generate_targets(min(CONFIG['batch_size'], len(data)))
    loss = loss_function(log_scores, targets)
    loss.backward()
    #grad clipping
    nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), CONFIG['norm_threshold'])
    optimizer.step()
    if i % 10 == 0:
        _LOGGER.info("batch: {:10d} | loss: {:.5f} | failed/skipped: {:4d}".format(i, loss, failed_or_skipped_batches))
        running_losses.append(loss.item())
        if len(running_losses) >  10:
            running_losses.pop(0)
    if i % 100 == 0:
        plotter.plot('loss', 'train', 'Loss', i, sum(running_losses) / len(running_losses))
    if i % 1000 == 0: 
        checkpoint_training(CONFIG['checkpoint_dir'], i, qt, optimizer)

checkpoint_training(CONFIG['checkpoint_dir'], -1, qt, optimizer, filename="FINAL_MODEL")            
end = time.time()
_LOGGER.info("Finished Training | Total Time: {}".format(end-start))