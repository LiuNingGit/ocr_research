# coding=utf-8
"""
@author: Ning Liu
---图片的预处理
"""
import cv2
import numpy as np
import os
from PIL import Image
from src.globalvar import GlobalVar
from src.img_template import OcrTemplate
from matplotlib import pyplot as plt
import src.img_template as it


# 图片预处理
def img_preprocess(filename):
    img = cv2.imread(filename)
    # 分离图片的噪声信息 和 核心信息
    noise_img, core_img = separate_by_hsv(img, OcrTemplate.noise_hsv_space)
    # 转换到空间
    noise_img = cv2.cvtColor(noise_img, cv2.COLOR_HSV2BGR)
    # 保存分离后的图片
    file_name_split = os.path.splitext(os.path.basename(filename))[0]
    noise_file_name = os.path.join(GlobalVar.get_temp_image_path(), file_name_split + '_noise.jpg')
    core_file_name = os.path.join(GlobalVar.get_temp_image_path(), file_name_split + '_core.jpg')
    save_img(noise_file_name, noise_img)
    save_img(core_file_name, core_img)
    return noise_file_name, core_file_name


# 根据模板从原始图片中分割出要识别的区域
def split_img(filename, template):
    noise_file_name, core_file_name = img_preprocess(filename)
    ocr_result = it.OcrResult(filename, template)
    # 计算投影矩阵
    img_core = cv2.imread(core_file_name, 0)
    img_verify = cv2.imread(template.verify_region.verify_template_name, 0)
    M, match_radio = img_match(img_verify, img_core, 0.3)
    ocr_result.match_radio = match_radio
    # 根据投影矩阵实现目标图片的变换提取
    img = cv2.imread(filename, 0)
    if M is not None:
        extract_recognition_region(img_core, M, ocr_result)
    else:
        return None
    return ocr_result


# 提取待识别的区域，并按编码保存
def extract_recognition_region(img, M, ocr_result):
    # 计算每块识别区域的相对坐标
    for recognition_rect, recognition_result in zip(OcrTemplate.recognition_region, ocr_result.recognition_region):
        code = recognition_rect.code
        x_r = recognition_rect.ul.x - OcrTemplate.verify_region.ul.x
        y_r = recognition_rect.ul.y - OcrTemplate.verify_region.ul.y
        w = recognition_rect.w
        h = recognition_rect.h
        img_rect = np.zeros((h, w, 3), np.uint8)
        pts_orig = np.float32([[y, x] for x in range(h) for y in range(w)]).reshape(-1, 1, 2)
        # 目标区域的坐标变换
        # img_rect = np.zeros((OcrTemplate.verify_region.h, OcrTemplate.verify_region.w, 1), np.uint8)
        # pts_orig = np.float32([[y, x] for x in range(OcrTemplate.verify_region.h) for y in range(OcrTemplate.verify_region.w)]).reshape(-1, 1, 2)
        pts = pts_orig + [y_r, x_r]
        dst = cv2.perspectiveTransform(pts, M)
        for p, d in zip(np.int32(pts_orig), np.int32(dst)):
            img_rect[p[0][1], p[0][0]] = img[d[0][1], d[0][0]]
        # show_image(img_rect)
        # 保存匹配到的区域图片
        file_name_split = os.path.splitext(os.path.basename(ocr_result.file_name))[0]
        img_name = os.path.join(GlobalVar.get_temp_image_path(), file_name_split + '_' + code + '.jpg')
        save_img(img_name, img_rect)
        raise_dpi(img_name)
        recognition_result.img_name = img_name
    # 保存截取后的图片


# 通过图像匹配检测目标图像中是否含有校验模板，并返回仿射矩阵
def img_match(query_img, train_img, min_match_radio=0.6):
    # 生成图片的sift描述符
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query_img, None)
    kp2, des2 = sift.detectAndCompute(train_img, None)

    # 比对两张图片特征，找到匹配的描述符
    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)
    min_dist = min(x.distance for x in matches)
    th_dist = max(min_dist * 2.0, 60.0)

    # 筛选匹配符距离比较小的描述符
    good = []
    for m in matches:
        if m.distance < th_dist:
            good.append(m)

    # 计算放射矩阵及其误差匹配
    if len(good) > min_match_radio * len(kp1):
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        match_radio = matches_mask.count(1) / len(kp1)
        # 通过画图可视化匹配结果
        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=None,
        #                    matchesMask=matches_mask,
        #                    flags=2)
        # img3 = cv2.drawMatches(query_img, kp1, train_img, kp2, good, None, **draw_params)
        # plt.imshow(img3, 'gray')
        # plt.show()
        # 返回结果
        if match_radio > min_match_radio:
            return M, match_radio
        else:
            return None, 0
    else:
        return None, 0


# 按颜色分离图片
def separate_by_hsv(img, hsv_space):
    # 转换到hsv空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    is_first = True
    for space in hsv_space:
        if is_first:
            mask = cv2.inRange(img_hsv, space.lower, space.upper)
            is_first = False
        else:
            mask |= cv2.inRange(img_hsv, space.lower, space.upper)
    img_mask = cv2.bitwise_and(img, img, mask=mask)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
    res = cv2.bitwise_or(img_binary, mask)
    return img_mask, res


# 显示图片
def show_image(img):
    res, pro = zoom_image(img, 800)
    cv2.imshow('image', res)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('messigray.png', img)
        cv2.destroyAllWindows()


# 增加图片dpi
def raise_dpi(filename, dpi=600):
    img = Image.open(filename)
    # dpi = img.info["dpi"]
    # img.info["dpi"] = dpi * k
    img.save(filename, dpi=(dpi, dpi))


# 缩放图片，并返回缩放系数
def zoom_image(img, target):
    w, h = img.shape[:2]
    real = max(w, h)
    pro = 1.0
    if real > target * 1.2:
        pro = float(target) / float(real)
        res = cv2.resize(img, None, fx=pro, fy=pro, interpolation=cv2.INTER_AREA)
    else:
        res = img
    return res, pro


# 保存图片
def save_img(img_name, img):
    img_path = os.path.dirname(img_name)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if os.path.exists(img_name):
        os.remove(img_name)
    cv2.imwrite(img_name, img)


if __name__ == '__main__':
    filename = r'E:\GitHub\ocr_research\template\4_2.jpg'
    noise_file_name, core_file_name = img_preprocess(filename)
    is_pass = img_split(core_file_name)
    print(is_pass)


