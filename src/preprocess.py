import logging
import time
from config import CONFIG
from data.utils import preprocess
from gensim.models import KeyedVectors

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

if __name__ == '__main__':
    WV_MODEL = KeyedVectors.load_word2vec_format(CONFIG['vec_path'], binary=True, limit=CONFIG['vocab_size'])
    _LOGGER.info("Starting data preprocessing")
    start = time.time()
    preprocess("{}/all.txt".format(CONFIG['base_dir']), "{}/cleaned.txt".format(CONFIG['base_dir']), WV_MODEL.vocab)
    end = time.time()
    _LOGGER.info("Took {:.2f} seconds to preprocess file".format(end-start))
