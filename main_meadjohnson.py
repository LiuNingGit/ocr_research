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
import src.img_template as it


if __name__ == '__main__':
    logger = lm.get_logger()
    try:
        result_str = ''
        #  filename = sys.argv[1]
        time_start = datetime.now()
        filename = r'E:\GitHub\ocr_research\template\4_2.jpg'
        # print('image path:%s' % filename)
        if not os.path.exists(filename):
            result_str = 'ocr_result:image not found '
        else:
            ocr_relsut = ocr.ocr_tesseract(filename, it.OcrTemplate)
            if ocr_relsut is not None:
                result_str = 'ocr_result:'
                for recognition_result in ocr_relsut.recognition_region:
                    result_str += '\n %s--%s \n' % (recognition_result.code, recognition_result.text)
            else:
                result_str = 'ocr_result:failed '
        time_end = datetime.now()
        second = (time_end - time_start).seconds
        result_str = result_str + ('total time:%ss' % second)
        print(result_str)
        # print('-----------------------------------------------------------------------------------------------')
        log_str = 'Image path:%s, %s' % (filename, result_str)
        logger.info(log_str)
    except:
        e = sys.exc_info()
        logger.info(e)
        # print('ErrorInfo:%s' % e)


