import torch
import torch.nn  as nn
import nltk
from gensim.utils import tokenize
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

data_path = '/home/jcaip/workspace/quickthoughts/bookcorpus/all.txt'
vec_path = '/home/jcaip/workspace/quickthoughts/S2V/GoogleNews-vectors-negative300.bin'

wv_model = KeyedVectors.load_word2vec_format(vec_path, binary=True, limit=None)

def prepare_sequence(text, vocab=wv_model.vocab):
    return torch.LongTensor([vocab[x].index for x in filter(lambda w: w in vocab, tokenize(text))])

class BookCorpus(data.dataset.Dataset):

    def __init__(self, file_path=data_path):
        print("Reading the data")
        with open(data_path, encoding='ISO-8859-1') as f:
            print("Creating Dataset")
            self.examples = list(f)
            print(len(self.examples))

    def __getitem__(self, i):
        return prepare_sequence(self.examples[i])

    def __len__(self):
        return len(self.examples)


class BOWEncoder(nn.Module):

    def __init__(self, embedding_filepath=vec_path, vocab=None):
        super(BOWEncoder, self).__init__()

        self.embeddings = nn.Embedding(*wv_model.vectors.shape)
        self.embeddings.weight = torch.nn.Parameter(torch.from_numpy(wv_model.vectors), 
                                                    requires_grad=False)

        self.w = nn.Linear(wv_model.vectors.shape[1], wv_model.vectors.shape[1])
                                                    

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        # print(embeds.shape)
        out = self.w(embeds)
        max, argmax = torch.max(out, dim=0)
        return max


class QuickThoughts(nn.Module):

    def __init__(self, encoder='bow'):
        super(QuickThoughts, self).__init__()

        if encoder =='bow':
            self.enc_f = BOWEncoder()

    def forward(self, inputs):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_f(inputs)

        # print("Encoding_f shape: {}".format(encoding_f.shape))
        # print("Encoding_g shape: {}".format(encoding_g.shape))

        scores = torch.matmul(encoding_f, encoding_g.t())

        # print("scores shape: {}".format(scores.shape))
        mask = torch.eye(len(scores)).byte()
        scores.masked_fill_(mask, 0)    

        return scores


bc = BookCorpus()
batch_size = 400
context_size = 1
train_iter = data.DataLoader(bc, batch_size=batch_size, num_workers=1, collate_fn=pad_sequence)

qt = QuickThoughts()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=5e-4)
loss_function = nn.BCEWithLogitsLoss()

print("Starting training")
for i, data in enumerate(train_iter):
    # print(i.text.shape)
    # print(data)
    a = qt(data)
    # print(a)
    # print(a.shape)

    # batch_size = min(batch_size, len(i.text))
    targets_np = np.zeros((batch_size, batch_size))
    ctxt_sent_pos = list(range(-context_size, context_size + 1))
    ctxt_sent_pos.remove(0)
    for ctxt_pos in ctxt_sent_pos:
        targets_np += np.eye(batch_size, k=ctxt_pos)

    targets_np_sum = np.sum(targets_np, axis=1, keepdims=True)
    targets_np = targets_np/targets_np_sum
    targets = torch.tensor(targets_np, dtype=torch.float)
    # print(targets)
    # print(targets.shape)

    loss = loss_function(a, targets)
    print("step {} loss: {}".format(i, loss))
    loss.backward()
    optimizer.step()
