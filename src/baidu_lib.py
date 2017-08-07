#!/usr/bin/env python

from aip import AipOcr
import urllib,sys
import ssl
import base64
import json


# 初始化百度服务id
APP_ID = '9753327'
API_KEY = 'ZddwHA2AC7cgsPGhMAO8Gadg'
SECRET_KEY = 'B8Vq9kIFVV0jq8cmCyIrRwbCsy6QZzGm'
Options = {
      'detect_direction': 'true',
      'language_type': 'CHN_ENG',
    }


# 读取图片
def get_file_content(filename):
    with open(filename, 'rb') as fp:
        return fp.read()


# 调用百度识别图片中的文字
def image_to_strings_by_sdk(filename):
    aipOcr = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    result = aipOcr.basicGeneral(get_file_content(filename), Options)
    for ss in result['words_result']:
        txt = txt + ss['words']
        txt = txt + '&'
    return txt


# 通过http调用百度api
def image_to_string_by_http(filename):
    token = AuthService()
    detectUrl = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    # 参数image：图像base64编码
    params = eval("{\"image\":" + "\"" + change_base(filename) + "\"" + "}")
    params = urllib.parse.urlencode(params).encode(encoding='utf8')
    detectUrl = detectUrl + "?access_token=" + token
    request = urllib.request.Request(url=detectUrl, data=params)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request)
    content = eval(response.read().decode('utf8'))
    txt = ''
    for ss in content['words_result']:
        txt = txt + ss['words']
    return txt


# 初始化http接口
def AuthService():
    # 获取token地址
    authHost = "https://aip.baidubce.com/oauth/2.0/token?"
    # 官网获取的 API Key
    clientId = "Hwy4IGyfG8NiafAkYP6hph3z"
    # 官网获取的 Secret Key
    clientSecret = "c3f8Q2y3H27R5CO5aDhsUOOfuWWLKOVv"
    getAccessTokenUrl = authHost + "grant_type=client_credentials" + "&client_id=" + clientId + "&client_secret=" + clientSecret
    response_data = urllib.request.urlopen(getAccessTokenUrl)
    params = json.loads(response_data.read())
    return params["access_token"]


# 转成base64
def change_base(filename):
    # 二进制方式打开图文件
    f = open(filename, 'rb')
    # 读取文件内容，转换为base64编码,但是还是二进制
    ls_f = base64.b64encode(f.read())
    f.close()
    return ls_f.decode()



