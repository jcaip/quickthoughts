from operator import itemgetter
from tqdm import tqdm

def prepare_sequence(text, vocab, max_len=50):
    pruned_sequence = zip(filter(lambda x: x in vocab, text), range(max_len))
    seq = [vocab[x].index for x, _ in pruned_sequence]
    return (seq, text)

#this function should process all.txt and removes all lines that are empty assuming the vocab
def preprocess(read_path, write_path, vocab, max_len=50):
    with open(read_path) as read_file, open(write_path, "w+") as write_file:
        # should all be iterators so fast
        write_file.writelines(line for _ , line in tqdm(filter(itemgetter(0),
                                                          map(lambda x: prepare_sequence(x, vocab, max_len=max_len),
                                                              read_file))))
