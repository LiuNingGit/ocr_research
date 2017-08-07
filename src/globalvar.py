# -*- coding:utf-8 -*-

"""
@author: Ning Liu
---全局配置文件
"""


class GlobalVar:
    """
    全局配置
    """
    # 本机
    tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    temp_image_path = r'E:\temp_image'
    log_path = r'E:\temp_image'
    template_image_path = r'E:\GitHub\ocr_research\template'
    # 阿里云服务器
    # tesseract_cmd = r'tesseract'
    # temp_image_path = r'/home/deploy/ocr_research/temp_image'
    # log_path = r'/home/deploy/ocr_research/temp_image'
    # template_image_path = r'/home/deploy/ocr_research/template'


    @staticmethod
    def get_tesseract_cmd():
        return GlobalVar.tesseract_cmd

    @staticmethod
    def get_temp_image_path():
        return GlobalVar.temp_image_path

    @staticmethod
    def get_log_path():
        return GlobalVar.log_path

    @staticmethod
    def get_template_image_path():
        return GlobalVar.template_image_path


