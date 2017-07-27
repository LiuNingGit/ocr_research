# -*- coding:utf-8 -*-

"""
@author: Ning Liu
---ocr的优化
"""
import os
import src.tesseract_lib as te
import src.generic_code as ge
import src.preprocessing as pr


def ocr_tesseract(image):
    tesseract_ocr = ''
    # Step1：预处理图片
    file_list = pr.split_image_hsv(image)
    # Step2: ocr图片
    for fl in file_list:
        fl_abs = os.path.abspath(fl)
        tesseract_ocr = tesseract_ocr + te.image_to_strings(fl_abs) + '&'
        os.remove(fl_abs)
    # Step3: 对结果进行修正
    tesseract_fix = ge.find_fix_no(tesseract_ocr)
    return tesseract_fix


