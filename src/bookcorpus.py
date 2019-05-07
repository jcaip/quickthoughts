from torch.utils.data.dataset import Dataset
from util import _WV_MODEL, prepare_sequence, base_dir, _LOGGER
import multiprocessing
import json
import torch

def prepare_sequence(text, vocab=_WV_MODEL.vocab, max_len=50, return_tensor=True):
    pruned_sequence = zip(filter(lambda x: x in vocab, text), range(max_len))
    seq = [vocab[x].index for x, _ in pruned_sequence]
    return seq


#this function  should  process all.txt and removes all lines that are empty assuming the vocab
def preprocess(file_path, write_path, vocab=_WV_MODEL.vocab, max_len=50):
    pool = multiprocessing.Pool(16)
    with open(file_path, encoding='ISO-8859-1') as read_file, open(write_path, "w+") as write_file:
        i, j= 0, 0
        for result in pool.imap(prepare_sequence, read_file):
            i+=1
            if len(result) != 0:
               j+=1
               write_file.write("{}\n".format(json.dumps(result)))

            if i % 100000 == 0:
                _LOGGER.info(json.dumps(result))
                _LOGGER.info("processed: {} wrote: {}".format(i, j))
    pool.close()

class BookCorpus(Dataset):
    def __init__(self, file_path, max_len=50):
        print("Reading the data")
        self.file_path=file_path
        with open(self.file_path, encoding='ISO-8859-1') as f:
            self.examples = list(f)

    def __getitem__(self, i):
        # this should be faster??
        return torch.LongTensor(json.loads(self.examples[i]))

    def __len__(self):
        #hack for now, hardcoded length
        return len(self.examples)


if __name__ == "__main__":
    _LOGGER.info("Working")
    preprocess("{}/all.txt".format(base_dir), "{}/cleaned.txt".format(base_dir))

