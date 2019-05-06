import torch
import torch.nn as nn
from util import wv_model

class Encoder(nn.Module):

    def __init__(self, wv_model, hidden_dim=1000):
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
        self.enc_f = Encoder(wv_model).cuda()
        self.enc_g = Encoder(wv_model).cuda()
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)

        scores = torch.matmul(encoding_f, encoding_g.t())
        mask = torch.eye(len(scores)).byte().cuda()
        scores.masked_fill_(mask, 0)    

        return self.softmax(scores)

