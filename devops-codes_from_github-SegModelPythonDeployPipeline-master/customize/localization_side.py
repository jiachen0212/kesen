import os

import cv2
import numpy as np

from utils import cv_imread_by_np, cv_imwrite
from utils import is_image
import time

def localize_one_edge(source_image, find_in_vertical=True, thre=None, expend=200):
    timestamp_start = time.perf_counter()
    if len(source_image.shape) == 2:
        source_image = source_image[:, :, None]

    h, w, c = source_image.shape
    if find_in_vertical:  # ver
        sample_numbers = 15
        sample_point = [int(i*w/sample_numbers) for i in range(sample_numbers)]
        sample_lines = source_image[:, sample_point, :]
        mean_max = np.max(np.mean(sample_lines, 1), 1)
        if thre is None:
            thre = np.mean(mean_max) * 0.8
        low_bound_max = h
    else:  # hor
        sample_numbers = 7
        sample_point = [int(i*h/sample_numbers) for i in range(sample_numbers)]
        sample_lines = source_image[sample_point, :, :]
        mean_max = np.max(np.max(sample_lines, 0), 1)
        if thre is None:
            thre = np.mean(sample_lines)*3
        low_bound_max = w
    candidate = np.where(mean_max > thre)
    up_bound = candidate[0][0] - expend
    low_bound = candidate[0][-1] + expend
    up_bound = 0 if up_bound < 0 else up_bound
    low_bound = low_bound_max if low_bound > low_bound_max else low_bound

    print(time.perf_counter() - timestamp_start)
    return up_bound, low_bound

def localize_for_side(source_image, thre=None, expend=200):
    top, bottom = localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
    left, right = localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
    return left, top, right, bottom


if __name__ == "__main__":
    # # 隧道光
    # dir_image = r"D:\Work\projects\kersen\data\0613-0614\0614-针眼-隧道【17】\隧道"
    # dir_out = r"D:\Work\projects\kersen\data\0613-0614\0614-针眼-隧道【17】\隧道demo_output"
    # func = localize_for_front_surface_Tunnel_Coaxial

    # # 同轴光
    # dir_image = r"D:\Work\projects\kersen\data\0610_previous\划伤\银白色\同轴"
    # dir_out = r"D:\Work\projects\kersen\data\0610_previous\划伤\银白色\同轴demo_output"
    # func = localize_for_front_surface_Tunnel_Coaxial

    # # # 分时频闪明场
    # dir_image = r"D:\Work\projects\kersen\data\0613-0614\0613-打磨异色【11】"
    # dir_out = r"D:\Work\projects\kersen\data\0613-0614\0613-打磨异色【11】demo_output"
    # func = localize_for_front_surface_Bar

    # # 侧面
    dir_image = r"D:\Work\projects\kersen\data\side\脏污异色-同轴"
    dir_out = r"D:\Work\projects\kersen\data\side\debug"
    func = localize_for_side

    for root, dirs, files in os.walk(dir_image):
        for file in files:
            if is_image(file):
                real_path = os.path.join(root, file)

                source_image = cv_imread_by_np(real_path)

                left, top, right, bottom = func(source_image, expend=100)
                print(left, top, right, bottom)

                pt1 = (left, top)
                pt2 = (right, bottom)
                cv2.rectangle(source_image, pt1, pt2, (255, 255, 255), 5)
                if not os.path.isdir(dir_out):
                    os.makedirs(dir_out)
                cv_imwrite(source_image, os.path.join(dir_out, file))
    print("DONE")
