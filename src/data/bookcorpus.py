import torch
import torch.nn  as nn
import nltk
from gensim.utils import tokenize
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from pprint import pprint

from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
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

data_path = '/home/jcaip/workspace/quickthoughts/bookcorpus/all.txt'
vec_path = '/home/jcaip/workspace/quickthoughts/S2V/GoogleNews-vectors-negative300.bin'

wv_model = KeyedVectors.load_word2vec_format(vec_path, binary=True, limit=10000)

def prepare_sequence(text, vocab=wv_model.vocab):
    tokens = list(tokenize(text))
    pruned = tokens[:min(50, len(tokens))]
    return torch.LongTensor([vocab[x].index for x in filter(lambda w: w in vocab, pruned)])

class BookCorpus(data.dataset.Dataset):

    def __init__(self, file_path=data_path):
        print("Reading the data")
        self.file_path=file_path

    def __getitem__(self, i):
        with open(self.file_path, encoding='ISO-8859-1') as f:
            for j, line in enumerate(f):
                if i == j:
                    return prepare_sequence(line)

    def __len__(self):
        #hack for now
        return 68196283


class Encoder(nn.Module):

    def __init__(self, embedding_filepath=vec_path, vocab=None, hidden_dim=1000):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_dim

        self.embeddings = nn.Embedding(*wv_model.vectors.shape)
        self.embeddings.weight = torch.nn.Parameter(torch.from_numpy(wv_model.vectors), 
                                                    requires_grad=False)
        self.gru = nn.GRU(wv_model.vectors.shape[1], hidden_dim)
                                                    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = torch.zeros(1, embeds.shape[1], self.hidden_size)
        if True:
            hidden = hidden.cuda()
        out, hidden = self.gru(embeds, hidden)

        return out[-1]

class QuickThoughts(nn.Module):

    def __init__(self, encoder='bow'):
        super(QuickThoughts, self).__init__()
        self.enc_f = Encoder().cuda()
        self.enc_g = Encoder().cuda()

    def forward(self, inputs):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)

        # print("Encoding_f shape: {}".format(encoding_f.shape))
        # print("Encoding_g shape: {}".format(encoding_g.shape))

        scores = torch.matmul(encoding_f, encoding_g.t())

        # print("scores shape: {}".format(scores.shape))
        mask = torch.eye(len(scores)).byte().cuda()
        scores.masked_fill_(mask, 0)    

        return scores

bc = BookCorpus()
batch_size = 400
context_size = 1
print("Reading the data")
with open(data_path, encoding='ISO-8859-1') as f:
    print("Creating Dataset")
    train_iter = data.DataLoader(bc, batch_size=batch_size, num_workers=10, collate_fn=pad_sequence)

qt = QuickThoughts()
for name, param in qt.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())

optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=5e-4)
loss_function = nn.BCEWithLogitsLoss(reduction='sum')

plotter = VisdomLinePlotter(env_name='qt')
norm_threshold=1.0 

print("Starting training")
for i, data in enumerate(train_iter):
    # print(i.text.j:shape)
    # print(data.shape)
    qt.zero_grad()
    data = data.cuda()
    a = qt(data)

    # print(a.shape)
    #TODO: Reformat this
    targets_np = np.zeros((batch_size, batch_size))
    ctxt_sent_pos = list(range(-context_size, context_size + 1))
    ctxt_sent_pos.remove(0)
    for ctxt_pos in ctxt_sent_pos:
        targets_np += np.eye(batch_size, k=ctxt_pos)

    targets_np_sum = np.sum(targets_np, axis=1, keepdims=True)
    targets_np = targets_np/targets_np_sum
    targets = torch.tensor(targets_np, dtype=torch.float).cuda()

    loss = loss_function(a, targets)
    plotter.plot('loss', 'train', 'Loss', i, loss.item())
    if i % 50 == 0:
        print("step {} loss: {}".format(i, loss))

        test_sentences =  [ "What is going on?",
                           "Let's go eat.",
                           "The engine won't start.",
                           "I'm hungry now."]

        pprint(test_sentences)

        test_sentences = pad_sequence(list(map(prepare_sequence, test_sentences))).cuda()
        print(qt(test_sentences))

    if i % 1000 == 0: 
        torch.save(qt.state_dict(), "../checkpoints/model-{}.pth".format(i))

    loss.backward()
    nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), norm_threshold)
    optimizer.step()
