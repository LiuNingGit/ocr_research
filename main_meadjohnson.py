# coding=utf-8
"""
@author: Ning Liu
---美赞臣ocr识别启动主文件
"""
import os
import sys
from datetime import datetime
import src.ocr as ocr
import src.log_manage as lm


if __name__ == '__main__':
    logger = lm.get_logger()
    try:
        result_str = ''
        filename = sys.argv[1]
        time_start = datetime.now()
        # filename = r'C:\Users\liuning\PycharmProjects\DemoOCR\picture\255.png'
        print('image path:%s' % filename)
        if not os.path.exists(filename):
            result_str = 'ocr_result:image not found '
        else:
            tesseract_fix, tesseract_ocr = ocr.ocr_tesseract(filename)
            if tesseract_fix:
                result_str = 'ocr_result:%s ' % tesseract_fix
            else:
                result_str = 'ocr_result:failed '
        time_end = datetime.now()
        second = (time_end - time_start).seconds
        result_str = result_str + (',total time:%ss' % second)
        print(result_str)
        print('-----------------------------------------------------------------------------------------------')
        log_str = 'Image path:%s, %s' % (filename, result_str)
        log_str_1 = 'Original ocr result:%s' % tesseract_ocr
        print(log_str_1)
        logger.info(log_str_1)
        logger.info(log_str)
    except:
        e = sys.exc_info()
        logger.info(e)
        print('ErrorInfo:%s' % e)


