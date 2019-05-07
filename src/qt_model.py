import torch
import torch.nn as nn
from util import _WV_MODEL



class Encoder(nn.Module):

    def __init__(self, wv_model, hidden_dim=1000):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_dim
        self.embeddings = nn.Embedding(*_WV_MODEL.vectors.shape)
        self.embeddings.weight = nn.Parameter(torch.from_numpy(wv_model.vectors), requires_grad=False)
        self.gru = nn.GRU(_WV_MODEL.vectors.shape[1], hidden_dim)
                                                    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = torch.zeros(1, embeds.shape[1], self.hidden_size).cuda()
        out, hidden = self.gru(embeds, hidden)

        masks = (vlens-1).view(1, -1, 1).expand(max_seq_len, outputs.size(1), outputs.size(2))
        out = out.gather(0, masks)[0]

        return out[-1]

class QuickThoughts(nn.Module):

    def __init__(self, encoder='bow'):
        super(QuickThoughts, self).__init__()
        self.enc_f = Encoder(wv_model)
        self.enc_g = Encoder(wv_model)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)

        scores = torch.matmul(encoding_f, encoding_g.t())
        # need to zero out when it's the same sentence
        mask = torch.eye(len(scores)).byte().cuda()
        scores.masked_fill_(mask, 0)    

        return self.softmax(scores)

