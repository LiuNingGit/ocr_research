# coding=utf-8

import os
import logging
from src.globalvar import GlobalVar


def get_logger():
    logger = logging.getLogger('[OCR]')
    dirpath = GlobalVar.get_log_path()
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    handler = logging.FileHandler(os.path.join(dirpath, "service.log"))
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger