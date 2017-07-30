#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: GuiLiang Zhu
---开源组件tesseract的调用封装
"""

import os
import subprocess
import tempfile
from src.globalvar import GlobalVar

# your tesseract exe file path
# tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
tesseract_cmd = GlobalVar.get_tesseract_cmd()


class TesseractError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)


def get_errors(error_string):
    '''
    returns all lines in the error_string that start with the string "error"

    '''

    error_string = error_string.decode('utf-8')
    lines = error_string.splitlines()
    error_lines = tuple(line for line in lines if line.find(u'Error') >= 0)
    if len(error_lines) > 0:
        return u'\n'.join(error_lines)
    else:
        return error_string.strip()


def tempnam():
    ''' returns a temporary file-name '''
    tmpfile = tempfile.NamedTemporaryFile(prefix="tess_")
    return tmpfile.name


def cleanup(filename):
    ''' tries to remove the given filename. Ignores non-existent files '''
    try:
        os.remove(filename)
    except OSError:
        pass


def run_tesseract(input_filename, output_filename_base, lang=None, psm=None, oem=None):

    command = [tesseract_cmd, input_filename, output_filename_base]

    if lang is not None:
        command += ['-l', lang]

    if psm is not None:
        command += ['--psm', psm]

    if oem is not None:
        command += ['--oem', oem]

    # command += ['digits']
    # print(command)
    proc = subprocess.Popen(command, stderr=subprocess.PIPE)
    status = proc.wait()
    error_string = proc.stderr.read()
    proc.stderr.close()
    return status, error_string


def image_to_strings(input_filename, lang=None, psm=None, oem=None):

    output_file_name_base = tempnam()

    output_file_name = '%s.txt' % output_file_name_base

    try:
        status, error_string = run_tesseract(input_filename, output_file_name_base, lang=lang, psm=psm, oem=oem)
        if status:
            errors = get_errors(error_string)
        with open(output_file_name, 'rb') as f:
            str = f.read().decode('utf-8').replace('\'', '').replace('\"', '').replace('\n', '').replace('”', '').replace('’', '').strip()
            return str

    finally:
        cleanup(output_file_name)