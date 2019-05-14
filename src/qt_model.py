import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import log_param_info

class UniGRUEncoder(nn.Module):

    def __init__(self, wv_model, hidden_size, cuda=True):
        super(UniGRUEncoder, self).__init__()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(*wv_model.vectors.shape)
        self.embeddings.weight = nn.Parameter(torch.from_numpy(wv_model.vectors), requires_grad=False)
        self.gru = nn.GRU(wv_model.vectors.shape[1], hidden_size)

    # input should be a packed sequence                                                    
    def forward(self, packed_input):
        #unpack to get the info we need
        raw_inputs, lengths = pad_packed_sequence(packed_input)
        max_seq_len = torch.max(lengths)
        embeds = self.embeddings(raw_inputs)
        hidden = torch.zeros(1, embeds.shape[1], self.hidden_size, device = self.device)
        packed = pack_padded_sequence(embeds, lengths, enforce_sorted=False)
        packed_output, hidden = self.gru(packed, hidden)
        unpacked, _ = pad_packed_sequence(packed_output)
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(1),
                                               unpacked.size(2)).unsqueeze(0).to(self.device)
        return unpacked.gather(0, idx).squeeze()

class QuickThoughts(nn.Module):

    def __init__(self, wv_model, hidden_size=1000, encoder='uni-gru', cuda=True):
        super(QuickThoughts, self).__init__()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.enc_f = UniGRUEncoder(wv_model, hidden_size, cuda=cuda)
        self.enc_g = UniGRUEncoder(wv_model, hidden_size, cuda=cuda)
        self.log_softmax = nn.LogSoftmax(dim=1)

    # generate targets softmax
    def generate_targets(self, num_samples):
        targets = torch.zeros(num_samples, num_samples, device=self.device)
        for offset in [-1, 1]:
            targets += torch.diag(torch.ones(num_samples-abs(offset), device=self.device), diagonal=offset)
        targets /= targets.sum(1, keepdim=True)
        return targets
        log_param_info(self)

    #expects a packed sequence
    def forward(self, inputs):
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)
        
        #testing
        if not self.training:
            return torch.cat((encoding_f, encoding_g), dim=1)

        #training
        scores = torch.matmul(encoding_f, encoding_g.t())
        # zero out when it's the same sentence
        mask = torch.eye(len(scores), device=self.device).byte()
        scores.masked_fill_(mask, 0)    

        #return log scores and target
        return self.log_softmax(scores)

    # generate target probability distribution
    def generate_targets(self, num_samples):
        targets = torch.zeros(num_samples, num_samples, device=self.device)
        for offset in [-1, 1]:
            targets += torch.diag(torch.ones(num_samples-abs(offset), device=self.device), diagonal=offset)
        targets /= targets.sum(1, keepdim=True)
        return targets
