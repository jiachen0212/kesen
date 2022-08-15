# coding=utf-8
import cv2
import json
import numpy as np
import os
from min_rec_locate import get_min_apple_pattern
from utils import cv_imread_by_np 

np.set_printoptions(suppress=True)

def perspective_transform(p: np.ndarray, m: np.ndarray):
    x, y = p
    u = (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / (m[2, 0] * x + m[2, 1] * y + m[2, 2])
    v = (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / (m[2, 0] * x + m[2, 1] * y + m[2, 2])
    return np.array([u, v])


def train1(conf: dict, point: list):

    # 记录train-image的最小logo矩形角点
    # 标注好的json记录好了apple-logo外围圈点集 
    p1, p2, p3, p4 =[point[2], point[1]], [point[2], point[3]], [0, point[3]], [0, point[1]]
    points = np.array([p1, p2, p3, p4], dtype='float32')
    mask_points = []
    for conf in conf["shapes"]:
        mask_ps = conf["points"]
        mask_ps = np.array(mask_ps)
        mask_points.append(mask_ps)

    return points, mask_points

# train 和 inference 利用的是: 不同图像中4个角点相对需mask点的位置都是一样的. 用反射矩阵求出四个角点的图像间的变换矩阵M,
# 则可以将这个M带入运算得到新图像中的mask点的坐标.
def inference1(image: np.ndarray, train_points: np.ndarray, train_mask_points: list, test_points: list):
    p1, p2, p3, p4 =[test_points[2], test_points[1]], [test_points[2], test_points[3]], [0, test_points[3]], [0, test_points[1]]
    points = np.array([p1, p2, p3, p4], dtype='float32')
    # 获得变换矩阵
    # train_points是train_data的四个角点, points则是新图像中学习到的4个角点. 做一个放射变换.
    m = cv2.getPerspectiveTransform(train_points, points)
    inference_mask_points = []
    # 注意这里使用的是: train_mask_points, 是train-data的conf-json我们mask好的点
    for ps in train_mask_points:
        inference_mask_ps = []
        for p in ps:
            # new_point就是新图像中我们该mask的点的位置了.
            new_point = perspective_transform(p, m)
            inference_mask_ps.append(new_point)
        inference_mask_ps = np.array(inference_mask_ps)
        inference_mask_points.append(inference_mask_ps)

    return points, inference_mask_points


def main1(test_dir):
 
    conf_path = r"D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\apple_logo_locate\fs_train_dir\fs_train.json"
    with open(conf_path) as f:
        train_conf = json.load(f)
    # 生成train-image的最小外接矩形坐标 
    train_dir = r'D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\apple_logo_locate\pattern_train'
    train_jiaodian = get_min_apple_pattern(train_dir)
    min_lapple_roi = train_jiaodian[list(train_jiaodian.keys())[0]]
    points, mask_points = train1(conf=train_conf, point=min_lapple_roi)

    # 对test-imag做logo-mask
    paths = [os.path.join(test_dir, a) for a in os.listdir(test_dir)]
    test_jiaodian = get_min_apple_pattern(test_dir)
    for cur_path in paths:
        print(cur_path)
        base_name = os.path.basename(cur_path)
        test_points = test_jiaodian[base_name]
        image = cv2.imread(cur_path, 0)
        draw = cv2.imread(cur_path)
        inference_points, inference_mask_points = inference1(image, points, mask_points, test_points)
        cv2.drawContours(draw, [inference_points.astype(np.int32)], 0, [0, 255, 0], 2)
        for ps in inference_mask_points:
            ps = ps.astype(np.int32)
            cv2.drawContours(draw, [ps], 0, [255, 255, 255], -1)
        # cv2.imshow("draw1", draw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(base_name, draw)

        draw[draw!=255] = 0

        return draw[:,:,0]




# if __name__ == '__main__':

def PerspectiveTransform(test_dir):
    # test_dir = r"D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\apple_logo_locate\fs_test_dir"
    masked = main1(test_dir)

    return masked

test_dir = r"D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\apple_logo_locate\fs_test_dir"
PerspectiveTransform(test_dir)