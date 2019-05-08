from torch.utils.data.dataset import Dataset
from util import _WV_MODEL, base_dir, _LOGGER
import multiprocessing
import torch
import time

def prepare_sequence(text, vocab=_WV_MODEL.vocab, max_len=50):
    pruned_sequence = zip(filter(lambda x: x in vocab, text), range(max_len))
    seq = [vocab[x].index for x, _ in pruned_sequence]
    return (seq, text)


#this function  should  process all.txt and removes all lines that are empty assuming the vocab
def preprocess(file_path, write_path, vocab=_WV_MODEL.vocab, max_len=50):
    pool = multiprocessing.Pool(8)
    with open(file_path) as read_file, open(write_path, "w+") as write_file:
        i, j= 0, 0
        for result, line in pool.imap(prepare_sequence, read_file):
            i+=1
            if result:
               write_file.write(line)
               j+=1

            if i % 1e6 == 0:
                percentage = 100*(i / 1e6) / 68.0
                _LOGGER.info("processed: {:2.2f}% wrote: {:10d} lines".format(percentage, j))
    pool.close()

class BookCorpus(Dataset):
    def __init__(self, file_path, max_len=50):
        _LOGGER.info("Reading the data")
        self.file_path=file_path
        with open(self.file_path, encoding='ISO-8859-1') as f:
            self.examples = list(f)

    def __getitem__(self, i):
        # this should be faster??
        return torch.LongTensor(prepare_sequence(self.examples[i])[0])

    def __len__(self):
        return len(self.examples)

if __name__ == '__main__':
    _LOGGER.info("Starting processing ...")
    start = time.time()
    preprocess("{}/all.txt".format(base_dir), "{}/cleaned.txt".format(base_dir))
    end = time.time()
    _LOGGER.info("Took {} time to preprocess file".format(end-start))
