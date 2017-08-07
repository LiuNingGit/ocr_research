# coding=utf-8
from collections import namedtuple

# 颜色定义
hsv_color = namedtuple('hsv_color', ('h', 's', 'v'))
# 颜色空间定义
hsv_color_space = namedtuple('hsv_color_space', ('lower', 'upper'))
# 定义像素坐标
pix = namedtuple('pix', ('x', 'y'))
# 定义验证矩形
verify_rect = namedtuple('verify_rect', ('ul', 'h', 'w', 'verify_template_name'))
# 定义识别矩形
recognition_rect = namedtuple('recognition_rect', ('code', 'ul', 'h', 'w',  'remark', 'lang'))


# 定义识别结果
class RecognitionResult:
    code = ''
    img_name = ''
    text = ''

    def __init__(self, code, img_name, text, lang):
        self.code = code
        self.img_name = img_name
        self.text = text
        self.lang = lang


# ocr 识别所用到的模板
class OcrTemplate:

    # 定义噪声颜色空间
    noise_hsv_space = [hsv_color_space(hsv_color(0, 43, 46), hsv_color(10, 255, 255)),
                       hsv_color_space(hsv_color(156, 43, 46), hsv_color(180, 255, 255))]
    # 定义验证区
    verify_region = verify_rect(pix(996, 1223), 239, 574, r'E:\GitHub\ocr_research\template\header.jpg')
    # 定义识别区
    recognition_region = [recognition_rect('recog1', pix(1590, 239), 400, 600, '货物或应税劳务、服务名称', 'chi_sim'),
                          recognition_rect('recog2', pix(2059, 858), 82, 944, '价税合计（大写）', 'chi_sim'),
                          recognition_rect('recog3', pix(2163, 1665), 217, 939, '货物或应税劳务、服务名称', 'chi_sim')]


# ocr 识别的结果
class OcrResult:
    # 匹配率
    match_radio = 0
    # 原始文件名
    file_name = ''

    def __init__(self, file_name, template):
        self.file_name = file_name
        self.recognition_region = [RecognitionResult(rect.code, '', '', rect.lang) for rect in template.recognition_region]





