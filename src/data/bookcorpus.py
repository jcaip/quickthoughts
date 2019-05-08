import logging
import torch
from torch.utils.data.dataset import Dataset
from data.utils import prepare_sequence


_LOGGER = logging.getLogger(__name__)

class BookCorpus(Dataset):

    def __init__(self, file_path, max_len=50):
        _LOGGER.info("Reading data from file: {}".format(file_path))
        with open(file_path, encoding='ISO-8859-1') as f:
            self.examples = list(f)
        _LOGGER.info("Successfully read {} lines".format(len(self.examples)))

    def __getitem__(self, i):
        return torch.LongTensor(prepare_sequence(self.examples[i])[0])

    def __len__(self):
        return len(self.examples)
