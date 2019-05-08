if __name__ == '__main__':
    _LOGGER.info("Starting data preprocessing")
    start = time.time()
    preprocess("{}/all.txt".format(CONFIG['base_dir']), "{}/cleaned.txt".format(CONFIG['base_dir']))
    end = time.time()
    _LOGGER.info("Took {:.2f} seconds to preprocess file".format(end-start))
