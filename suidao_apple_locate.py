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

    flag = 3
    # flag = 2 对腐蚀得到的定位apple-logo做放大, 实现apple-logo外扩10mm

    if flag == 0:
        # 隧道的右半张子图  [apple-logo少的那半张]
        sd2_image = cv2.imread(r'D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir\A180_KSA0000000879003_Snow_Station4_Linear_Tunnel_2_2022_08_29_16_01_19_196_RC_N_Ori.bmp')
        sd2_image = cv2.cvtColor(sd2_image, cv2.COLOR_BGR2RGB) 
        sd2_image = cv2.cvtColor(sd2_image, cv2.COLOR_RGB2GRAY)  
        a, b = localize_one_edge(sd2_image, find_in_vertical=True, thre=None, expend=200)
        c, d = localize_one_edge(sd2_image, find_in_vertical=False, thre=None, expend=200)
        print(a, b, c, d)
        # d是物料的右边界, 隧道右子图的孔和竖线, 相对右边界距离是固定的(除非相机的分辨率变了.). 这里固定-3900就ok
        d -= 3900
        # a是物料上边界, 固定+1000充分剔除冗余且不至于影响apple-logo;
        # b是物料下边界, 固定-1000充分剔除冗余且不至于影响apple-logo;
        a += 8000
        b -= 8000
        wuliao_tz = sd2_image[a:b, c:d]
        cv2.imwrite('./wuliao_sd.jpg', wuliao_tz) 
        otsuThe, maxValue = 0, 255  # otsuThe=46
        _, dst_Otsu = cv2.threshold(wuliao_tz, otsuThe, maxValue, cv2.THRESH_OTSU)
        dst_Otsu = cv2.bitwise_not(dst_Otsu)

        timestamp_start = time.perf_counter()
        # 检测出的apple-logo边上还是有点黑点, so先腐蚀(去除黑点)再膨胀(外扩白像素.)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
        dst_Otsu1 = cv2.erode(dst_Otsu, kernel, iterations=1)
        # full_mask_ = np.zeros_like(sd2_image)
        # full_mask_[a:b, c:d] = dst_Otsu1
        # cv2.imwrite('./eroded_apple_mask.jpg', full_mask_)

        # # 0901, 多加几次膨胀, 让apple-logo外阔多一些
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
        dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 2) 
        for i in range(10):
            dst_Otsu2 = cv2.dilate(dst_Otsu2, kernel, 2)  
            # print(np.sum(dst_Otsu2))
        print("10次膨胀", time.perf_counter() - timestamp_start)  # 2.4s

        # 大核外扩 
        # kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(130,130))
        # dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel_large, 2)  
        
        # print("大核膨胀", time.perf_counter() - timestamp_start)  # 1.8s 

        # 把abcd添加回去, 得到和原图一样size的apple-mask
        full_mask = np.zeros_like(sd2_image)
        full_mask[a:b, c:d] = dst_Otsu2
        print(full_mask.shape)
        cv2.imwrite('./2.jpg', full_mask)

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
        _, dst_Otsu = cv2.threshold(wuliao_tz, otsuThe, maxValue, cv2.THRESH_OTSU)
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
    
    elif flag == 2:
        import math
        half_logo_length = 1043
        base_img = cv2.imread('./eroded_apple_mask.jpg')  # h,w: 22000 8192
        h, w = base_img.shape[:2]
        first = 0
        while not base_img[first][0][0]: 
            first += 1
        apple_center = half_logo_length+first
        erode_mask = cv2.resize(base_img, (int(w*1.5), int(h*1.5)), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('./erode_mask_mask.jpg', erode_mask)
        first = 0
        while not erode_mask[first][0][0]: 
            first += 1
        apple_center1 = first + math.ceil(half_logo_length*1.5) 
        diff_ = apple_center1 - apple_center
        modif_erode_mask = erode_mask[diff_:diff_+h, :w]
        show_merged_mask = cv2.addWeighted(modif_erode_mask, 0.5, base_img, 0.5, 10)     
        cv2.imwrite('./q.jpg', show_merged_mask)
    
    elif flag == 3:
        img1 = cv2.imread('./test_apple_mask.jpg')
        img2 = cv2.imread(r'D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir\A180_KSA0000000879003_Snow_Station4_Linear_Tunnel_2_2022_08_29_16_01_19_196_RC_N_Ori.bmp')
        # stemp = cv2.bitwise_not(img2)
        # cv2.imwrite('./stemp.jpg', stemp)
        show_merged_mask = cv2.addWeighted(img1, 0.2, img2, 0.8, 10)     
        cv2.imwrite('./3.jpg', show_merged_mask)
        
        
