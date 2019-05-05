import torch
import torch.nn  as nn
import nltk
from gensim.utils import tokenize
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

data_path = '/home/jcaip/workspace/quickthoughts/bookcorpus/all.txt'
vec_path = '/home/jcaip/workspace/quickthoughts/S2V/GoogleNews-vectors-negative300.bin.gz'

def prepare_sequence(text, vocab):
    return torch.LongTensor(map(lambda x: vocab[x].index, filter(lambda w: w in vocab, tokenize(text))))

class BookCorpus(data.dataset.Dataset):

    def __init__(self, file_path=data_path):
        print("Reading the data")
        with open(data_path, encoding='ISO-8859-1') as f:
            print("Creating Dataset")
            examples_list = map(lambda x: prepare_sequence(text, vocab), fields), f)

    def __getitem__(i):

    print("Building Vocab!")

    def 


class BOWEncoder(nn.Module):

    def __init__(self, embedding_filepath=vec_path, vocab=None):
        super(BOWEncoder, self).__init__()

        self.wv_model = KeyedVectors.load_word2vec_format(embedding_filepath, binary=True, limit=vocab)
        self.embeddings = nn.Embedding(*self.wv_model.vectors.shape)
        self.embeddings.weight.data.copy_(vocab.vectors)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        # print(embeds.shape)
        max, argmax = torch.max(embeds, dim=0)
        return max


class QuickThoughts(nn.Module):

    def __init__(self,vocab,  encoder='bow'):
        super(QuickThoughts, self).__init__()

        if encoder =='bow':
            self.enc_f = BOWEncoder(vocab)
            self.enc_g = BOWEncoder(vocab)

    def forward(self, inputs):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)

        # print("Encoding_f shape: {}".format(encoding_f.shape))
        # print("Encoding_g shape: {}".format(encoding_g.shape))

        scores = torch.matmul(encoding_f, encoding_g.t())

        # print("scores shape: {}".format(scores.shape))
        mask = torch.eye(len(scores)).byte()
        scores.masked_fill_(mask, 0)    

        return scores


    batch_size = 64
    context_size = 1
    train_iter = data.Iterator(bc, batch_size=batch_size)

    qt = QuickThoughts(TEXT.vocab)
    optimizer = optim.Adam(qt.parameters(), lr=5e-4)
    loss_function = nn.BCEWithLogitsLoss()

    print("Starting training")
    for i, data in enumerate(train_iter):
        # print(i.text.shape)
        a = qt(data.text)
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
