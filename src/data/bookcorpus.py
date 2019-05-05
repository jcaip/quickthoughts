import torch
import torch.nn 
import nltk
from gensim.utils import tokenize
from torchtext.data.dataset import Dataset
import torchtext.data as data

data_path = '/home/jcaip/workspace/quickthoughts/bookcorpus/head.txt'

def tokenizer(text):
    return [tok for tok in tokenize(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer)
fields=[('Text', TEXT)]

print("Reading the data")
with open(data_path, encoding='ISO-8859-1') as f:
    bc = Dataset([data.Example.fromlist([x.strip()], fields) for x in f if x.strip() != ''], fields=fields)

TEXT.build_vocab(bc, vectors="glove.6B.100d")
train_iter = data.Iterator(bc, batch_size=64)

for i in train_iter:
    print(i)
    print(type(i))
    print(i.Text)
    # vocab = TEXT.vocab
    # self.embed = nn.Embedding(len(vocab), emb_dim)
    # self.embed.weight.data.copy_(vocab.vectors)
