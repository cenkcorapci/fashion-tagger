# -*- coding: utf-8 -*-
"""Model configs.
"""

import logging
import pathlib

# Logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
LOGS_PATH = '/tmp/tb_logs/'

# Experiments
RANDOM_STATE = 41

# Local files
ROOT_PATH = '/run/media/twoaday/data-storag/data-sets/fashion-product-images-small/'
STYLES_DATA_SET_PATH = ROOT_PATH + 'styles.csv'
IMAGES_FOLDER_PATH = ROOT_PATH + 'images/'
PRE_TRAINED_FASHION_TAGGER_WEIGHTS = None
DL_MODELS_PATH = ROOT_PATH + 'models/dl/'
TB_LOGS_PATH = ROOT_PATH + 'tb_logs/'

CATEGORY_COUNTS_DICT = {'subCategory': 45,
                        'articleType': 142,
                        'gender': 5,
                        'season': 4,
                        'baseColour': 46,
                        'usage': 8}

# create directories
logging.info("Checking directories...")
pathlib.Path(DL_MODELS_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")
