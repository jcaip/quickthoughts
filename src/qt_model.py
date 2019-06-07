import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformer_model import PositionalEncoding, PositionwiseFeedForward, MultiHeadedAttention, Encoder, EncoderLayer
from utils import log_param_info
from scipy.linalg import block_diag
import numpy as np

class GRUEncoder(nn.Module):
    def __init__(self, wv_model, hidden_size, bidirectional, dropout,  cuda=True):
        super(GRUEncoder, self).__init__()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(*wv_model.vectors.shape)
        self.embeddings.weight = nn.Parameter(torch.from_numpy(wv_model.vectors))
        self.bidirectional = bidirectional
        self.gru = nn.GRU(wv_model.vectors.shape[1], hidden_size, dropout=dropout, bidirectional=bidirectional)

    # input should be a packed sequence                                                    
    def forward(self, packed_input):
        #unpack to get the info we need
        raw_inputs, lengths = pad_packed_sequence(packed_input)
        max_seq_len = torch.max(lengths)
        embeds = self.embeddings(raw_inputs)
        hidden = torch.zeros(2 if self.bidirectional else 1, embeds.shape[1], self.hidden_size, device = self.device)
        packed = pack_padded_sequence(embeds, lengths, enforce_sorted=False)
        packed_output, hidden = self.gru(packed, hidden)
        unpacked, _ = pad_packed_sequence(packed_output)
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(1),
                                               unpacked.size(2)).unsqueeze(0).to(self.device)
        return unpacked.gather(0, idx).squeeze()


class TransformerEncoder(nn.Module):
    """
    Attention based sentence encoder
    """
    def __init__(self, wv_model, hidden_size, cuda=True, N=6, d_model=300, d_ff=1000, h=5, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.embeddings = nn.Embedding(*wv_model.vectors.shape)
        self.embeddings.weight = nn.Parameter(torch.from_numpy(wv_model.vectors), requires_grad=False)
        self.position = PositionalEncoding(d_model, dropout)
        self.src_embed = nn.Sequential(self.embeddings, self.position)
        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, self.attn, self.ff, dropout), N)

        
    def forward(self, packed_input, pad=0):
        padded_input, lengths = pad_packed_sequence(packed_input, batch_first=True)
        # print("padded input:", padded_input.size())
        mask = (padded_input != pad).unsqueeze(-2)
        temp = self.src_embed(padded_input)
        # print("temp:", temp.size())
        unpacked = self.encoder(temp, mask)
        # print("encoded:", unpacked.size())

        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1).to(self.device)
        # print("idx:", idx.size())
        output = unpacked.gather(1, idx).squeeze()
        # print("output:", output.size())
        return output

class QuickThoughts(nn.Module):

    def __init__(self, wv_model, hidden_size=1000, encoder='uni-gru', cuda=True):
        super(QuickThoughts, self).__init__()
        self.device = torch.device('cuda' if cuda else 'cpu')
        # self.enc_f = TransformerEncoder(wv_model, hidden_size, cuda=cuda)
        # self.enc_g = TransformerEncoder(wv_model, hidden_size, cuda=cuda)
        self.enc_f = GRUEncoder(wv_model, hidden_size, False, 0.3,  cuda=cuda)
        self.enc_g = GRUEncoder(wv_model, hidden_size, False, 0.3, cuda=cuda)
        log_param_info(self)

    # generate targets softmax
    def generate_targets(self, num_samples, offsetlist=[1], label_smoothing=0.1):
        targets = torch.zeros(num_samples, num_samples, device=self.device).fill_(label_smoothing)
        for offset in offsetlist:
            targets += torch.diag(torch.ones(num_samples-abs(offset), device=self.device), diagonal=offset)
        targets /= targets.sum(1, keepdim=True)
        return targets


    # generate batched targets
    def generate_block_targets(self, positive_block_size, num_blocks):
        # positive_labels = np.ones((positive_block_size, positive_block_size)) - np.eye(positive_block_size)
        positive_labels = np.ones((positive_block_size, positive_block_size)) - np.eye(positive_block_size)
        np_targets = block_diag(*([positive_labels] * num_blocks))
        targets = torch.from_numpy(np_targets).to(self.device)
        return targets

    def generate_smooth_targets(self, num_samples):
        targets = torch.zeros(num_samples, num_samples, device=self.device)
        for offset, scale in zip([-3, -2, -1, 1, 2, 3], [1, 1, 10, 10, 1, 1]):
            targets += scale*torch.diag(torch.ones(num_samples-abs(offset), device=self.device), diagonal=offset)
        targets /= targets.sum(1, keepdim=True)
        return targets

    #expects a packed sequence
    def forward(self, inputs, catdim=1):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)
        
        #testing
        if not self.training:
            return torch.cat((encoding_f, encoding_g), dim=catdim)

        return (encoding_f, encoding_g)

