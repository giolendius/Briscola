import logging
from pathlib import Path


DATA_FOLDER = Path.cwd() / 'data'
dataset_file_name = 'dataset.csv'


def setup_logger(name) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(4)
    logging.basicConfig(level=4)
    return logger