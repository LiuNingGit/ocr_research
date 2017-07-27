# coding=utf-8
"""
@author: Ning Liu
---美赞臣ocr识别启动主文件
"""
import sys
import src.ocr as ocr
import os


if __name__ == '__main__':
    try:
        filename = sys.argv[1]
        #  filename = r'C:\Users\liuning\PycharmProjects\DemoOCR\picture\255.jpeg'
        if not os.path.exists(filename):
            print('ErrorInfo:ocr的图片不存在！')
        else:
            tesseract_fix = ocr.ocr_tesseract(filename)
            if tesseract_fix:
                print('SuccessInfo:%s' % tesseract_fix)
            else:
                print('ErrorInfo:未识别到出生编码！')
    except:
        e = sys.exc_info()
        print('ErrorInfo:%s' % e)
