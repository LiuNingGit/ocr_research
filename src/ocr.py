# -*- coding:utf-8 -*-

"""
@author: Ning Liu
---ocr的优化
"""
import os
import src.tesseract_lib as te
import src.preprocessing as pr
import src.baidu_lib as baidu


def ocr_tesseract(filename, template):
    tesseract_ocr = ''
    # Step1：预处理图片
    ocr_relsut = pr.split_img(filename, template)
    if ocr_relsut is None:
        return ocr_relsut
    # Step2: ocr图片
    for recognition_result in ocr_relsut.recognition_region:
        fl_abs = os.path.abspath(recognition_result.img_name)
        # recognition_result.text = te.image_to_strings(fl_abs, recognition_result.lang)
        recognition_result.text = baidu.image_to_string_by_http(fl_abs)
        # print(tesseract_ocr)
        # os.remove(fl_abs)
    # Step3: 对结果进行修正
    # tesseract_fix = ge.find_fix_no(tesseract_ocr)
    # print(tesseract_fix)
    return ocr_relsut


