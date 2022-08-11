import os
import time

import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import cv_imread_by_np, cv_imwrite


def localize_one_edge(source_image, find_in_vertical=True, thre=None, expend=200):
    # timestamp_start = time.perf_counter()
    if len(source_image.shape) == 2:
        source_image = source_image[:, :, None]

    h, w, c = source_image.shape
    if find_in_vertical:  # ver
        sample_point = (int(w * 3 / 7), int(w * 1 / 2), int(w * 4 / 7))
        sample_lines = source_image[:, sample_point, :]
        mean_max = np.max(np.mean(sample_lines, 1), 1)
        if thre is None:
            thre = np.mean(mean_max) * 0.8
        low_bound_max = h
    else:  # hor
        sample_point = (int(h * 3 / 7), int(h * 1 / 2), int(h * 4 / 7))  # avoid center logo
        sample_lines = source_image[sample_point, :, :]
        mean_max = np.max(np.max(sample_lines, 0), 1)
        if thre is None:
            thre = np.mean(mean_max)
        low_bound_max = w
    candidate = np.where(mean_max > thre)
    up_bound = candidate[0][0] - expend
    low_bound = candidate[0][-1] + expend
    up_bound = 0 if up_bound < 0 else up_bound
    low_bound = low_bound_max if low_bound > low_bound_max else low_bound

    # print(time.perf_counter() - timestamp_start)
    return up_bound, low_bound


def get_masked_subimg(sub_image, mask_bright=True, tresh=100, expend=10,max_radius=35):
    # 将图片转为灰度图
    if len(sub_image.shape) > 2:
        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = sub_image

    retval, dst = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)
    if not mask_bright:
        dst = 255 - dst
    _, labels = cv2.connectedComponents(dst, 8)
    max_size = -1
    max_mask = None
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = labels == label
        target_size = np.sum(label_mask)

        if target_size > max_size:
            max_size = target_size
            max_mask = label_mask

    if max_mask is not None:
        contours, _ = cv2.findContours(max_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1
        x, y, w, h = cv2.boundingRect(contours[0])
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        center = (center_x, center_y)
        r = min(int(max(w, h) / 2), max_radius) + expend
        print(r)
        max_mask = np.ones_like(max_mask).astype(np.uint8)
        cv2.circle(max_mask, center, r, 0, -1)
        if len(sub_image.shape) == 3:
            max_mask = max_mask[:, :, None]
        """
            For 德成：用这个max_mask将mic空区域所有的缺陷都屏蔽（模型输出缺陷图，后处理之前进行掩除），不做输出。
        """
        return max_mask * sub_image
    else:
        return sub_image


def localize_for_front_surface_Tunnel_Coaxial(source_image, thre=None, expend=200):
    top, bottom = localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
    left, right = localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
    return left, top, right, bottom


def localize_for_front_surface_Bar(source_image, thre=None, expend=200):
    top, bottom = localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
    left, right = localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
    cal_bottom = top + 19500
    bottom = min(cal_bottom, bottom)
    return left, top, right, bottom


def localize_for_front_surface_mic_Tunnel_Coaxial(source_image, thre=None, expend=0):
    top, bottom = localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
    left, right = localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
    # 孔相对物料的边界固定, 复制固定的值即可. [除非相机分辨率变了.]
    y_center = int(top + 9565)
    x_center = int(left + 330)
    # print(bottom - top)
    # 孔的半径
    r = 50
    left = x_center - r
    top = y_center - r
    right = x_center + r
    bottom = y_center + r
    source_image[top:bottom, left:right] = get_masked_subimg(source_image[top:bottom, left:right], False, tresh=50,
                                                             expend=0, max_radius=18)
    return left, top, right, bottom


def localize_for_front_surface_mic_Bar(source_image, thre=None, expend=200):
    top, bottom = localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
    left, right = localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
    # 找到右边边界, 然后mic孔之类相对边界的距离其实是固定的[除非相机分辨率变了.] 
    # so直接加上固定的值即可, 就固定到了圆心坐标
    y_center = int(top + 9565)
    x_center = int(left + 330)
    r = 150
    # 设置mic孔的半径为150
    left = x_center - r
    top = y_center - r
    right = x_center + r
    bottom = y_center + r
    source_image[top:bottom, left:right] = get_masked_subimg(source_image[top:bottom, left:right], expend=10,max_radius=35)

    return left, top, right, bottom


if __name__ == "__main__":

    # kersen station 1 mic Coaxial
    # dir_image = r"D:\BaiduNetdiskDownload\out_0805\1_position\2"
    # dir_out = r"D:\BaiduNetdiskDownload\out_0805\1_position\2_mic_res"
    # func = localize_for_front_surface_mic_Tunnel_Coaxial

    # # kersen station 3 mic Bar
    # dir_image = r"D:\BaiduNetdiskDownload\out_0805\3_position\1"
    # dir_out = r"D:\BaiduNetdiskDownload\out_0805\3_position\1_mic_res"
    # func = localize_for_front_surface_mic_Bar

    # for root, dirs, files in os.walk(dir_image):
    #     for file in files:

    #         real_path = os.path.join(root, file)
    #         source_image = cv_imread_by_np(real_path)

    #         # timestamp_start = time.perf_counter()
    #         left, top, right, bottom = func(source_image, expend=0)
    #         # print(time.perf_counter() - timestamp_start)

    #         print(left, top, right, bottom)

    #         pt1 = (left, top)
    #         pt2 = (right, bottom)
    #         cv2.rectangle(source_image, pt1, pt2, (255, 255, 255), 5)
    #         if not os.path.isdir(dir_out):
    #             os.makedirs(dir_out)
    #         expend = 100
    #         source_image = source_image[top-expend:bottom+expend,left-expend:right+expend]
    #         cv_imwrite(source_image, os.path.join(dir_out, file))
    # print("DONE")

    
    # Fenshi
    # image_path = r'D:\work\project\DL\kesen\data\fs.bmp'
    # image = np.asarray(Image.open(image_path))
    # cv2.imwrite('./fs.jpg', image)

    image_path = './fs.jpg'
    image = cv_imread_by_np(image_path)
    a, b = localize_one_edge(image, find_in_vertical=True, thre=None, expend=200)
    c, d = localize_one_edge(image, find_in_vertical=False, thre=None, expend=200)
    print(a,b,c,d)
    # b是底部的泛光冗余, 相对位置固定(除非相机分辨率改变), 直接数值剔除
    b -= 1500
    # 左上都剔除一些冗余, 可固定写si
    a += 1000
    c += 1000
    # apple-logo相对有边界的巨鹿固定, 数值可写si
    d -= 3600
    image = image[a:b, c:d]
    cv2.imwrite('./fs_wuliao.jpg', image)
    # otsuThe, maxValue = 0, 255  # otsuThe=46
    # otsuThe, dst_Otsu = cv2.threshold(image, 1, maxValue, cv2.THRESH_OTSU)
    # dst_Otsu = cv2.bitwise_not(dst_Otsu)
    # # 检测出的apple-logo边上还是有点黑点, so先腐蚀(去除黑点)再膨胀(外扩白像素.)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
    # dst_Otsu1 = cv2.erode(dst_Otsu, kernel, iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
    # dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 5)  

    # # 存apple-logo的mask图.
    # cv2.imwrite('./fs.jpg', dst_Otsu2)
