from torch.utils.data.dataset import Dataset
from util import _WV_MODEL, prepare_sequence
import multiprocessing

#this function  should  process all.txt and removes all lines that are empty assuming the vocab
def preprocess(file_path, write_path, vocab=_WV_MODEL.vocab, max_len=50):
    pool = multiprocessing.Pool(10)
    with open(file_path, encoding='ISO-8859-1') as read_file, open(write_path) as write_file:
        i, j= 0, 0
        for result in p.imap(prepare_sequence, read_file)
            i++
            if len(result) != 0:
               j++
               write_file.write(result)
               print("processed: {} wrote: {}".format(i, j))
    pool.close()

class BookCorpus(Dataset):
    def __init__(self, file_path, max_len=50):
        print("Reading the data")
        self.file_path=file_path
        with open(self.file_path, encoding='ISO-8859-1') as f:
            self.examples = (f)

    def __getitem__(self, i):
        return prepare_sequence(self.examples[i])

    def __len__(self):
        #hack for now, hardcoded length
        return len(self.examples)

