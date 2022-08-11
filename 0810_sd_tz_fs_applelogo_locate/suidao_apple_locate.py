# coding=utf-8

import os
import time
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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



if __name__ == "__main__":

    flag = 1 # 左半图
    # flag = 0  # 右半图

    if flag == 0:
        # 隧道的右半张子图  [apple-logo少的那半张]
        sd2_image = cv2.imread(r'D:\work\project\DL\kesen\data\19-2-8.bmp')
        sd2_image = cv2.cvtColor(sd2_image, cv2.COLOR_BGR2RGB) 
        sd2_image = cv2.cvtColor(sd2_image, cv2.COLOR_RGB2GRAY)  
        a, b = localize_one_edge(sd2_image, find_in_vertical=True, thre=None, expend=200)
        c, d = localize_one_edge(sd2_image, find_in_vertical=False, thre=None, expend=200)
        print(a, b, c, d)
        # d是物料的右边界, 隧道右子图的孔和竖线, 相对右边界距离是固定的(除非相机的分辨率变了.). 这里固定-3900就ok
        d -= 3900
        # a是物料上边界, 固定+1000充分剔除冗余且不至于影响apple-logo;
        # b是物料下边界, 固定-1000充分剔除冗余且不至于影响apple-logo;
        a += 1000
        b -= 1000
        wuliao_tz = sd2_image[a:b, c:d]
        otsuThe, maxValue = 0, 255  # otsuThe=46
        otsuThe, dst_Otsu = cv2.threshold(wuliao_tz, otsuThe, maxValue, cv2.THRESH_OTSU)
        
        dst_Otsu = cv2.bitwise_not(dst_Otsu)
        # 检测出的apple-logo边上还是有点黑点, so先腐蚀(去除黑点)再膨胀(外扩白像素.)
        kernel = np.ones((30, 30), dtype=np.uint8)
        dst_Otsu1 = cv2.erode(dst_Otsu, kernel, iterations=1)
        kernel = np.ones((50, 50), dtype=np.uint8)
        dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 5)  
        # 把abcd添加回去, 得到和原图一样size的apple-mask
        full_mask = np.zeros_like(sd2_image)
        full_mask[a:b, c:d] = dst_Otsu2
        print(full_mask.shape)
        cv2.imwrite('./sd2_apple_mask.jpg', full_mask)

    elif flag == 1:
        # 隧道的左半张子图   [apple-logo多的那半张]
        sd2_image = cv2.imread(r'D:\work\project\DL\kesen\data\sd1.bmp')
        sd2_image = cv2.cvtColor(sd2_image, cv2.COLOR_BGR2RGB) 
        sd2_image = cv2.cvtColor(sd2_image, cv2.COLOR_RGB2GRAY)  
        a, b = localize_one_edge(sd2_image, find_in_vertical=True, thre=None, expend=200)
        c, d = localize_one_edge(sd2_image, find_in_vertical=False, thre=None, expend=200)
        # print(a, b, c, d)
        # c是物料的左边界, 隧道左子图的孔相对左边界距离是固定的(除非相机的分辨率变了.). 这里固定+100就ok
        c += 3000
        # a是物料上边界, 固定+1000充分剔除冗余且不至于影响apple-logo;
        # b是物料下边界, 固定-1000充分剔除冗余且不至于影响apple-logo;
        a += 8000
        b -= 8000
        wuliao_tz = sd2_image[a:b, c:d]
        cv2.imwrite('./wuliao_tz.jpg', wuliao_tz)
        otsuThe, maxValue = 0, 255  # otsuThe=63
        otsuThe, dst_Otsu = cv2.threshold(wuliao_tz, otsuThe, maxValue, cv2.THRESH_OTSU)
        _, dst_Otsu = cv2.threshold(wuliao_tz, otsuThe+20, maxValue, cv2.THRESH_OTSU)
        dst_Otsu = cv2.bitwise_not(dst_Otsu)
        # 检测出的apple-logo边上还是有点黑点, so先腐蚀(去除黑点)再膨胀(外扩白像素.)
        # kernel = np.ones((30, 30), dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
        dst_Otsu1 = cv2.erode(dst_Otsu, kernel, iterations=1)
        # kernel = np.ones((50, 50), dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
        dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, iterations=2)  
        # 把abcd添加回去, 得到和原图一样size的apple-mask
        full_mask = np.zeros_like(sd2_image)
        full_mask[a:b, c:d] = dst_Otsu2
        # print(full_mask.shape)
        cv2.imwrite('./sd1_apple_mask.jpg', full_mask)