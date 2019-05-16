import itertools
from tqdm import tqdm
from gensim.utils import tokenize
  
def prepare_sequence(text, vocab, max_len=50):
    pruned_sequence = itertools.islice(filter(lambda x: x in vocab, tokenize(text)), max_len)
    seq = [vocab[x].index for x in pruned_sequence]
    return seq

#this function should process all.txt and removes all lines that are empty assuming the vocab
def preprocess(read_path, write_path, vocab, max_len=50):
    #get the length
    with open(read_path) as read_file:
        file_length = sum(1 for line in read_file)

    with open(read_path) as read_file, open(write_path, "w+") as write_file:
        write_file.writelines(tqdm(filter(lambda x: prepare_sequence(x, vocab, max_len=max_len), read_file), total=file_length))

import logging
import time
from config import CONFIG
from data.utils import preprocess
from gensim.models import KeyedVectors
import gensim.downloader as api

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

if __name__ == '__main__':
    WV_MODEL = api.load('glove-wiki-gigaword-300')
    _LOGGER.info("Starting data preprocessing")
    start = time.time()
    preprocess("{}/data/all.txt".format(CONFIG['base_dir']), "{}/data/cleaned.txt".format(CONFIG['base_dir']), WV_MODEL.vocab)
    end = time.time()
    _LOGGER.info("Took {:.2f} seconds to preprocess file".format(end-start))