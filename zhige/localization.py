# coding=utf-8
'''
zhige 的投影法.

'''
import os
import time

import cv2
import numpy as np

from utils import cv_imread_by_np, cv_imwrite
from utils import is_image


def localize_one_edge(source_image, find_in_vertical=True, thre=80, expend=50):
    # timestamp_start = time.perf_counter()

    h, w, c = source_image.shape
    if find_in_vertical:  # ver
        sample_point = (int(w * 3 / 7), int(w * 1 / 2), int(w * 4 / 7))
        sample_lines = source_image[:, sample_point, :]
        mean_max = np.max(np.mean(sample_lines, 1), 1)
        low_bound_max = h
    else:  # hor
        sample_point = (int(h * 2 / 7), int(h * 1 / 2), int(h * 5 / 7))  # avoid center logo
        sample_lines = source_image[sample_point, :, :]
        mean_max = np.max(np.mean(sample_lines, 0), 1)
        low_bound_max = w
    candidate = np.where(mean_max > thre)
    up_bound = candidate[0][0] - expend
    low_bound = candidate[0][-1] + expend
    up_bound = 0 if up_bound < 0 else up_bound
    low_bound = low_bound_max if low_bound > low_bound_max else low_bound

    # print(time.perf_counter() - timestamp_start)
    return up_bound, low_bound


def localize_one_item(source_image, thre=80, expend=50):
    top, bottom = localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
    left, right = localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
    return left, top, right, bottom


if __name__ == "__main__":
    # # 隧道光
    # dir_image = r"D:\Work\projects\kersen\kersen\0613-0614\0614-针眼-隧道【17】\隧道"
    # dir_out = r"D:\Work\projects\kersen\kersen\0613-0614\0614-针眼-隧道【17】\隧道demo_output"

    # 同轴光
    dir_image = r"D:\Work\projects\kersen\kersen\all_before_0610\划伤\银白色\同轴"
    dir_out = r"D:\Work\projects\kersen\kersen\all_before_0610\划伤\银白色\同轴demo_output"

    # 分时频闪明场    目前失效  考虑定位到左右上点后直接计算下点在哪里
    # dir_image = r"D:\Work\projects\kersen\kersen\0613-0614\0613-打磨异色【11】"
    # dir_out = r"D:\Work\projects\kersen\kersen\0613-0614\0613-打磨异色【11】demo_output"

    for root, dirs, files in os.walk(dir_image):
        for file in files:
            if is_image(file):
                real_path = os.path.join(root, file)

                source_image = cv_imread_by_np(real_path)
                left, top, right, bottom = localize_one_item(source_image)

                pt1 = (left, top)
                pt2 = (right, bottom)
                cv2.rectangle(source_image, pt1, pt2, (255, 255, 255), 100)
                cv_imwrite(source_image, os.path.join(dir_out, file))

    print("DONE")
