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
    preprocess("{}/all.txt".format(CONFIG['base_dir']), "{}/cleaned.txt".format(CONFIG['base_dir']), WV_MODEL.vocab)
    end = time.time()
    _LOGGER.info("Took {:.2f} seconds to preprocess file".format(end-start))
