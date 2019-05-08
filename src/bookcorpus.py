import logging
import multiprocessing
import time
import torch
from operator import itemgetter
from torch.utils.data.dataset import Dataset
from util import WV_MODEL 
from config import CONFIG

_LOGGER = logging.getLogger(__name__)

def prepare_sequence(text, vocab=WV_MODEL.vocab, max_len=50, return_original=False):
    pruned_sequence = zip(filter(lambda x: x in vocab, text), range(max_len))
    seq = [vocab[x].index for x, _ in pruned_sequence]
    return (seq, text)

#this function should process all.txt and removes all lines that are empty assuming the vocab
def preprocess(read_path, write_path, vocab=WV_MODEL.vocab, max_len=50):
    pool = multiprocessing.Pool(8)
    with open(read_path) as read_file, open(write_path, "w+") as write_file:
        # should all be iterators so fast
        write_file.writelines(line for _ , line in filter(itemgetter(0),
                                                          pool.imap(prepare_sequence, read_file)))
    pool.close()

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

if __name__ == '__main__':
    _LOGGER.info("Starting data preprocessing")
    start = time.time()
    preprocess("{}/all.txt".format(CONFIG['base_dir']), "{}/cleaned.txt".format(CONFIG['base_dir']))
    end = time.time()
    _LOGGER.info("Took {:.2f} seconds to preprocess file".format(end-start))
