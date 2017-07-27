# coding=utf-8
"""
@author: Ning Liu
---图片的预处理
"""
import cv2
import numpy as np
import os
import copy
from PIL import Image
from src.globalvar import GlobalVar

# 定义绿色空间
LOWER_GREEN = np.array([45, 20, 23])   # 100张图片时效果比较好时的参数
UPPER_GREEN = np.array([88, 255, 255])


# 预处理图片
def split_image_hsv(filename):
    img = cv2.imread(filename)
    img_org = copy.deepcopy(img)
    # 缩放图片
    img, pro = zoom_image(img, 700)
    # 转换成hsv空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lg = LOWER_GREEN
    ug = UPPER_GREEN
    # 根据绿色空间过滤图片
    img_hsv_mask = mask_image(img_hsv, lg, ug)
    # 灰度化
    img_mask_bgr = cv2.cvtColor(img_hsv_mask, cv2.COLOR_HSV2BGR)
    img_mask_gray = cv2.cvtColor(img_mask_bgr, cv2.COLOR_BGR2GRAY)
    # 二值化
    # plt.hist(img_mask_gray)
    # img_mask_gray_thresh = np.where(img_mask_gray > 20, 253, 0)
    ret, img_mask_gray_thresh = cv2.threshold(img_mask_gray, 30, 255, cv2.THRESH_BINARY_INV)
    img_mask_gray_thresh_fill = fill_hollow(img_mask_gray_thresh, 4)
    # 将边缘全部设置为白色
    img_mask_gray_border = white_border(img_mask_gray_thresh_fill, 5)
    # 识别轮廓
    image, contours, hierarchy = cv2.findContours(img_mask_gray_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 循环处理轮廓
    i_c = 0
    file_list = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        h_org, w_org, r_org = img_org.shape
        org_area = h_org * w_org
        # 绘制轮廓
        # img_cnt = cv2.drawContours(img_org, cnt, -1, (0, 0, 255), 3)  # Draw contours
        # show_image(img_cnt)
        if org_area / 10 > cnt_area > 256:
            # 根据轮廓截取图片，并增强后保存
            ((x_m, y_m), (w_m, h_m), th_m) = cv2.minAreaRect(cnt)
            w_m_p = w_m / pro
            h_m_p = h_m / pro
            if max(w_m, h_m) / min(w_m, h_m) > 3.2 or max(w_m, h_m) / min(w_m, h_m) < 2.3 or max(w_m, h_m) > 200 or min(w_m_p, h_m_p) < 35:
                # print('舍弃的矩形长:%s 宽:%s' % (max(w_m, h_m), min(w_m, h_m)))
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x_p = int(x / pro); y_p = int(y / pro); w_p = int(w / pro); h_p = int(h / pro);
            img_cut = copy.deepcopy(img_org[y_p:(y_p + h_p), x_p:(x_p + w_p)])
            # 通过g,r_g分布二值化图片
            img_depoint_list = thresh_otsu(img_cut)
            # 保存原始图片
            # img_depoint_list.append(img_cut)

            path = GlobalVar.get_temp_image_path()
            if not os.path.exists(path):
                os.makedirs(path)
            for j_c in range(0, len(img_depoint_list)):
                fn = r'%s\%s_split_%s_%s.jpg' % (path, os.path.splitext(os.path.basename(filename))[0], i_c, j_c)
                if os.path.exists(fn):
                    os.remove(fn)
                cv2.imwrite(fn, img_depoint_list[j_c])
                if os.path.exists(fn):
                    raise_dpi(fn)
                file_list.append(fn)
                j_c += 1
            i_c += 1
    return file_list


# 缩放图片，并返回缩放系数
def zoom_image(img, target):
    w, h = img.shape[:2]
    real = max(w, h)
    pro = 1.0
    if real > target * 1.2:
        pro = float(target) / float(real)
        img = cv2.resize(img, None, fx=pro, fy=pro, interpolation=cv2.INTER_AREA)
    return img, pro


# 根据hsv范围过滤图片
def mask_image(img, lower, upper):
    mask = cv2.inRange(img, lower, upper)
    img_mask = cv2.bitwise_and(img, img, mask=mask)
    return img_mask


# 填充二值化图片里的空洞
def fill_hollow(img_bin, k):
    img_bin_copy = img_bin.copy()
    w, h = img_bin.shape[:2]
    th = (2 * k + 1) * (2 * k + 1) * 175
    # 边缘点不进行填充
    for y in range(1, h - 2):
        for x in range(1, w - 2):
            if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                continue
            x_min = (x - k) if x - k > 0 else 0
            x_max = (x + k) if x + k < w - 1 else (w - 1)
            y_min = (y - k) if y - k > 0 else 0
            y_max = (y + k) if y + k < h - 1 else (h - 1)
            bin_sum = np.sum(img_bin_copy[x_min: x_max, y_min: y_max])
            if bin_sum < th:
                img_bin[x, y] = 0
            else:
                img_bin[x, y] = 255
    return img_bin


# 设置图片边缘为白色
def white_border(img, k):
    w, h = img.shape[:2]
    img[0:k, :] = 255
    img[:, 0:k] = 255
    img[w-k-1:w, :] = 255
    img[:, h-k-1:h] = 255
    return img


# 自适应阀值对图片进行二值化
def thresh_otsu(img):
    img_org = copy.deepcopy(img)
    g = copy.deepcopy(img[:, :, 1]).astype(np.int32)
    r = copy.deepcopy(img[:, :, 2]).astype(np.int32)
    b = copy.deepcopy(img[:, :, 0]).astype(np.int32)
    gray = (r * 30 + g * 59 + b * 11)/100
    r_g = r - g
    mean_gray = np.mean(gray)
    median_gray = np.median(gray)
    std_gray = np.std(gray)
    mean_r_g = np.mean(r_g)
    median_r_g = np.median(r_g)
    std_r_g = np.std(r_g)
    mean_g = np.mean(g)
    std_g = np.std(g)
    img_list = []
    for i in range(1, 10, 2):
        for j in range(1, 10, 2):
            for k in range(1, 10, 2):
                th_g = mean_g - std_g * (i + 5) / 10
                th_gray = mean_gray - std_gray * j / 10
                th_r_g = mean_r_g - std_r_g * k / 10
                # print('th_g %s-----' % th_g, 'th_r_g %s-----' % th_r_g)
                img_temp = copy.deepcopy(img_org)
                image_binary(img_temp, th_g, th_r_g, th_gray)
                # img_temp_enlarge = enlarge(img_temp)
                img_temp_enlarge_blur = cv2.bilateralFilter(img_temp, 5, 75, 25)
                img_list.append(img_temp_enlarge_blur)
    return img_list


# 二值化图片
def image_binary(img, th_g, th_r_g, th_gray):
    img_mask = copy.deepcopy(img).astype('float')
    g, r, b =img_mask[:, :, 1], img_mask[:, :, 2], img_mask[:, :, 0]
    r_g = r - g
    gray = img_mask[:, :, 2] * 0.3 + img_mask[:, :, 1] * 0.59 + img_mask[:, :, 0] * 0.11
    black = (g < th_g) & (r_g >= th_r_g) & (gray < th_gray)
    white = ~ black
    img[black] = (0, 0, 0)
    img[white] = (255, 255, 255)


# 放大图片
def enlarge(img):
    h, w = img.shape[:2]
    if w * 3 < 900 and h * 3 < 600:
        img_enlarge = cv2.resize(img, (w * 3, h * 3))
        return img_enlarge
    else:
        return img


# 增加图片dpi
def raise_dpi(filename, dpi=600):
    img = Image.open(filename)
    # dpi = img.info["dpi"]
    # img.info["dpi"] = dpi * k
    img.save(filename, dpi=(dpi, dpi))
