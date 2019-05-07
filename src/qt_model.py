import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import _WV_MODEL
class Encoder(nn.Module):

    def __init__(self, wv_model, hidden_dim=1000):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_dim
        self.embeddings = nn.Embedding(*_WV_MODEL.vectors.shape)
        self.embeddings.weight = nn.Parameter(torch.from_numpy(wv_model.vectors), requires_grad=False)
        self.gru = nn.GRU(_WV_MODEL.vectors.shape[1], hidden_dim)

    # input should be a packed sequence                                                    
    def forward(self, packed_input):
        #unpack to get the info we need
        raw_inputs, lengths = pad_packed_sequence(packed_input)
        max_seq_len = torch.max(lengths)

        embeds = self.embeddings(raw_inputs)
        hidden = torch.zeros(1, embeds.shape[1], self.hidden_size).cuda()

        packed = pack_padded_sequence(embeds, lengths, enforce_sorted=False)
        packed_output, _ = self.gru(packed, hidden)
        output , _ = pad_packed_sequence(packed_output)

        masks = (lengths-1).unsqueeze(0).unsqueeze(2).expand(max_seq_len, output.size(1), output.size(2)).cuda()
        last_outputs = output.gather(0, masks)[0]

        return last_outputs

    def last_timestep(self, unpacked, lengths):
        """last_timestep
        computes the index of the last timestep to get the output
        :param unpacked:
        :param lengths:
        """
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(1),
                                               unpacked.size(2)).unsqueeze(0).cuda()
        return unpacked.gather(0, idx).squeeze()
class QuickThoughts(nn.Module):

    def __init__(self, encoder='bow'):
        super(QuickThoughts, self).__init__()
        self.enc_f = Encoder(_WV_MODEL)
        self.enc_g = Encoder(_WV_MODEL)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)

        scores = torch.matmul(encoding_f, encoding_g.t())
        # need to zero out when it's the same sentence
        mask = torch.eye(len(scores)).byte().cuda()
        scores.masked_fill_(mask, 0)    

        return self.softmax(scores)

