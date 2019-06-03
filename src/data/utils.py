import itertools
from tqdm import tqdm
from gensim.utils import tokenize
  
def prepare_sequence(text, vocab, max_len=50, no_zeros=False):
    pruned_sequence = zip(filter(lambda x: x in vocab, tokenize(text)), range(max_len))
    seq = [vocab[x].index for (x, _) in pruned_sequence]
    if len(seq) == 0 and no_zeros:
        return [1]
    return seq

#this function should process all.txt and removes all lines that are empty assuming the vocab
def preprocess(read_path, write_path, vocab, max_len=50):
    #get the length
    with open(read_path) as read_file:
        file_length = sum(1 for line in read_file)

    with open(read_path) as read_file, open(write_path, "w+") as write_file:
        write_file.writelines(tqdm(filter(lambda x: prepare_sequence(x, vocab, max_len=max_len), read_file), total=file_length))

