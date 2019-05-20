import logging
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from data.utils import prepare_sequence

_LOGGER = logging.getLogger(__name__)
class BookCorpus(Dataset):

    def __init__(self, file_path, vocab,  max_len=50, block_size=-1):
        self.block_size = block_size
        self.vocab = vocab
        with open(file_path) as f:
            if self.block_size > 0:	
                self.examples = np.array_split(f, block_size)
            else:
                self.examples = list(f)

        self.num_sampled_from_chunk = 0
        self.chunk_idx_offsets = list(range(0, len(self), 


        _LOGGER.info("Successfully read {} lines from file: {}".format(len(self.examples), file_path))

    def __getitem__(self, i):
        if self.block_size > 0:
            torch.LongTensor(prepare_sequence(self.examples[i], self.vocab))
            if self.num_sampled_from_chunk 

            
        else:
            torch.LongTensor(prepare_sequence(self.examples[i], self.vocab))

    def __len__(self):
        return len(self.examples)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
