from torch.utils.data.dataset import Dataset
from util import _WV_MODEL, prepare_sequence
from multiprocessing import Pool

#this function  should  process all.py
def preprocess():
    pass

class BookCorpus(Dataset):

    def __init__(self, file_path, max_len=50):
        print("Reading the data")
        self.file_path=file_path
        pool = Pool(processes=10)
        with open(self.file_path, encoding='ISO-8859-1') as f:
            self.examples = list(f)

    def __getitem__(self, i):
        return prepare_sequence(self.examples[i])

    def __len__(self):
        #hack for now, hardcoded length
        return len(self.examples)

