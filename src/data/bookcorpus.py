import logging
import torch
from torch.utils.data.dataset import Dataset
from data.utils import prepare_sequence

_LOGGER = logging.getLogger(__name__)
class BookCorpus(Dataset):

    def __init__(self, file_path, vocab,  max_len=50):
        self.vocab = vocab
        with open(file_path) as f:
            self.examples = list(f)
        _LOGGER.info("Successfully read {} lines from file: {}".format(len(self.examples), file_path))

    def __getitem__(self, i):
        return torch.LongTensor(prepare_sequence(self.examples[i], self.vocab))

    def __len__(self):
        return len(self.examples)
