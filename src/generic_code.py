# -*- coding:utf-8 -*-
"""
@author: Ning Liu
---通用代码
"""
import re
from collections import Counter


#  正则找出出生证编号
def find_no(text):
    strlist = re.findall('([P|Q|R]\d{9})', text, re.I)
    return ','.join(strlist)


def fix_code(strings):
    re = ''
    if len(strings) <= 0:
        return re
    ls = strings.split(',')
    for i in range(10):
        number = Counter([j[i] for j in ls if len(j) == 10])
        if len(number) == 0:
            continue
        mode = max(number.items(), key=lambda x: x[1])[0]
        if i == 0 and mode == '0':
            mode = 'Q'
        re += mode

    return re


def find_fix_no(text):
    strlist = re.findall('([0|P|Q|R]\d{9})', text, re.I)
    str_no = ','.join(strlist)
    no = fix_code(str_no.upper())
    return no

