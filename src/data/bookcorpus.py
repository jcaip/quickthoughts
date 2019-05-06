
from torch.utils.data.dataset import Dataset
from ../util.py import wv_model

class BookCorpus(Dataset):

    def __init__(self, file_path=data_path, max_len=50):
        print("Reading the data")
        self.file_path=file_path
        self.max_len = max_len


    def __getitem__(self, i):
        with open(self.file_path, encoding='ISO-8859-1') as f:
            for j, line in enumerate(f):
                if i == j:
                    return self.prepare_sequence(line)

    def __len__(self):
        #hack for now, hardcoded length
        return 68196283

    def prepare_sequence(self, text, vocab=wv_model.vocab):
        tokens = list(tokenize(text))
        pruned = tokens[:min(self.max_len, len(tokens))]
        return torch.LongTensor([vocab[x].index for x in filter(lambda w: w in vocab, pruned)])
