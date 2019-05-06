from torch.utils.data.dataset import Dataset
from util import wv_model, prepare_sequence

# class BookCorpus(Dataset):

    # def __init__(self, file_path, max_len=50):
        # print("Reading the data")
        # self.file_path=file_path
        # self.max_len = max_len


    # def __getitem__(self, i):
        # with open(self.file_path, encoding='ISO-8859-1') as f:
            # for j, line in enumerate(f):
                # if i == j:
                    # return prepare_sequence(line)

    # def __len__(self):
        # #hack for now, hardcoded length
        # return 68196283

class BookCorpus(Dataset):

    def __init__(self, file_path, max_len=50):
        print("Reading the data")
        self.file_path=file_path
        with open(self.file_path, encoding='ISO-8859-1') as f:
            self.examples = list(f)



    def __getitem__(self, i):
        return prepare_sequence(self.examples[i])

    def __len__(self):
        #hack for now, hardcoded length
        return len(self.examples)

