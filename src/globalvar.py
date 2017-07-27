# -*- coding:utf-8 -*-

"""
@author: Ning Liu
---全局配置文件
"""


class GlobalVar:
    """
    全局配置
    """
    tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    temp_image_path = r'E:\temp_image'

    @staticmethod
    def get_tesseract_cmd():
        return GlobalVar.tesseract_cmd

    @staticmethod
    def get_temp_image_path():
        return GlobalVar.temp_image_path


