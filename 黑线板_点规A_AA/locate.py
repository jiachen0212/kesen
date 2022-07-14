
import json
import os
from re import L
import sys
from turtle import left

from torch import zeros
import cv2
import json
import numpy as np
cv2.CV_IO_MAX_IMAGE_PIXELS = 200224000
from PIL import Image
import numpy as np
import cv2
cv2.CV_IO_MAX_IMAGE_PIXELS = 200224000
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(image, five_holes_area, roi):
    m = cv2.matchTemplate(image, five_holes_area, cv2.TM_CCORR_NORMED)
    ys, xs = np.where(m == m.max())
    x, y = int(xs[0]), int(ys[0])
    area_points = [[x, y],
                   [x + five_holes_area.shape[1], y],
                   [x + five_holes_area.shape[1], y + five_holes_area.shape[0]],
                   [x, y + five_holes_area.shape[0]]]

    m = cv2.matchTemplate(image, roi, cv2.TM_CCORR_NORMED)
    ys, xs = np.where(m == m.max())
    left_top = (int(xs[0]), int(ys[0]))

    train_info = {
        # "roi": roi,
        "area_points": area_points,
        "mark_point": left_top,
        "score": float(m.max())
    }

    return train_info


def inference(image, train_info, verbose=False):
    roi = train_info.get("roi")
    area_points = train_info.get("area_points")
    area_points = np.array(area_points)
    mark_point = train_info.get("mark_point")
    
    m = cv2.matchTemplate(image, roi, cv2.TM_CCORR_NORMED)
    ys, xs = np.where(m == m.max())
    x, y = xs[0], ys[0]

    # 1. area_points - mark_point 新图像中的冗余在减法中被抵消;
    # 2. 新图像中template的坐标找到, 加到1.的结果上, 则得到完整五孔[area_points]的坐标啦.
    new_area_points = area_points - mark_point + (x, y)
    if verbose:
        draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        draw = cv2.drawContours(draw, [new_area_points], 0, (0, 255, 0), 2)
    else:
        draw = None
    inference_info = {
        "score": float(m.max()),
        "area_points": new_area_points.tolist(),
        # "draw": draw
    }
    return inference_info


def cv_imread_by_np(filePath, clr_type=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), clr_type)

    return cv_img


def cv_imwrite(image, dst):
    name = os.path.basename(dst)
    _, postfix = os.path.splitext(name)[:2]
    cv2.imencode(ext=postfix, img=image)[1].tofile(dst)


def apple_circle(zero_mask, center, r):
    # 只在圆心+-r范围内判断点和圆的相对位置关系
    left_up = [int(center[0]-r), int(center[1]-r)]
    right_bottom = [int(center[0]+r), int(center[1]+r)]
    right = min(right_bottom[0]+1, zero_mask.shape[1])
    for i in range(left_up[0], right):
        for j in range(left_up[1], right_bottom[1]+1):
                if (i-center[0])**2 + (j-center[1])**2  <= r**2:
                    zero_mask[j][i] = 255

def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def main_fun(js_dir, test_path, train_dir=None, flag=None, roi_vis_path=None):
    ims = os.listdir(test_path)
    paths = [os.path.join(test_path, a) for a in ims if '.bmp' in a]

    # 开启 train: left and right   
    # img = cv2.imread(os.path.join(train_dir, "image.bmp"), 0)
    # five_holes_area = cv2.imread(os.path.join(train_dir, "imac.bmp"), 0)
    # roi = cv2.imread(os.path.join(train_dir, "template.bmp"), 0)
    # train_info = train(img, five_holes_area, roi)
    # with open(os.path.join(train_dir, "train_info.json"), "w") as f:
    #     json.dump(train_info, f, indent=4)
    
    # 开启 train: apple logo   
    # img = cv2.imread(os.path.join(train_dir, "image.jpg"), 0)
    # five_holes_area = cv2.imread(os.path.join(train_dir, "apple.jpg"), 0)
    # roi = cv2.imread(os.path.join(train_dir, "apple_template.jpg"), 0)
    # train_info = train(img, five_holes_area, roi)
    # with open(os.path.join(train_dir, "apple.json"), "w") as f:
    #     json.dump(train_info, f, indent=4)

    # inference 
    if flag == 'left':
        with open(os.path.join(js_dir, "left.json")) as f:
            train_info = json.load(f)
    elif flag == 'right':
        with open(os.path.join(js_dir, "right.json")) as f:
            train_info = json.load(f)
    elif flag == 'apple_logo':
        with open(os.path.join(js_dir, "apple.json")) as f:
            train_info = json.load(f)
        ttemp = os.path.join(train_dir, "apple_template.jpg")
        roi = cv2.imread(ttemp, 0)
        train_info["roi"] = roi
        for img_path in paths:
            im_name = os.path.basename(img_path)
            img = cv_imread_by_np(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inference_info = inference(img, train_info, verbose='False')
            _, p2 = inference_info['area_points'][0], inference_info['area_points'][2]
            # cv2.circle(img, (int(p2[0]), int(p2[1])), 10, (0, 255, 0), 20)
            # cv2.imwrite('./3.jpg', img)
            print(p2)
            temp_point = [4697, 10197]
            r = np.sqrt((temp_point[0]-p2[0])**2 + (temp_point[1]-p2[1])**2)
            temp_point1 = [4497, 9997]
            r1 = np.sqrt((temp_point1[0]-p2[0])**2 + (temp_point1[1]-p2[1])**2)
            print(r, r1)
            # points = [p2] + [temp_point] + [temp_point1]
            # cv2.circle(img, (int(p2[0]), int(p2[1])), int(r), (255, 255, 255), 20)
            # cv2.circle(img, (int(p2[0]), int(p2[1])), int(r1), (255, 255, 255), 20)
            # cv2.imwrite('./3.jpg', img)
            zero_mask = np.zeros_like(img)
            apple_circle(zero_mask, p2, r1)
            cv2.imwrite('./{}_apple_logo_mask.jpg'.format(im_name.split('.')[0]), zero_mask)

            return zero_mask

    ttemp = os.path.join(train_dir, "template_{}.bmp".format(flag))
    roi = cv2.imread(ttemp, 0)
    train_info["roi"] = roi
    img_roi = dict()
    for img_path in paths:
        im_name = os.path.basename(img_path)
        img = cv_imread_by_np(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inference_info = inference(img, train_info, verbose='False')
        p1, p2 = inference_info['area_points'][0], inference_info['area_points'][2]
        roi = p1 + p2
        img_roi[im_name] = roi

        if roi_vis_path:
            cv2.rectangle(img, p1, p2, (255, 255, 255), 10, 8)
            cuted_dirfix, _ = os.path.splitext(im_name)[:2]
            temp_path =  os.path.join(roi_vis_path, test_path.split('\\')[-2])
            mkdir(temp_path)
            cv_imwrite(img, os.path.join(temp_path, cuted_dirfix+'.jpg'))

    return img_roi

    # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # # 剔除冗余, 把边上的黑色部分剔除, 只剩下干净的物料
    # image = image[1185:19892, 2500: 8192]

    # # 二值化并找到apple-logo的最小外接圆
    # otsuThe, maxValue = 0, 255
    # # otsuThe, dst_Otsu = cv2.threshold(image, otsuThe, maxValue, cv2.THRESH_OTSU)
    # # print(otsuThe)
    # _, dst_Otsu = cv2.threshold(image, 0, maxValue, cv2.THRESH_OTSU)
    # dst_Otsu = cv2.bitwise_not(dst_Otsu)
    # cv2.imwrite('./1.jpg', dst_Otsu)
    # contours, _ = cv2.findContours(dst_Otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]

    # # (x, y), radius = cv2.minEnclosingCircle(cnt)  
    # # print((x, y), radius)
    # # # cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 20)
    # # cv2.circle(image, (int(x),int(y)), 10, (255,255,255), 10)
    # # cv2.imwrite('./1.jpg', image)
    # contours, _ = cv2.findContours(dst_Otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 绘制轮廓
    # cv2.drawContours(dst_Otsu, contours, -1, (128,128,128), 30, lineType=cv2.LINE_AA)
    # cv2.imwrite('./2.jpg', dst_Otsu)

    # img_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\apple\image.jpg'
    # image = cv2.imread(img_path)
    # # temple = image[920:10593, 1560:6553]
    # temple = image[920:15491, 1560:7537]
    # cv2.imwrite('./apple_template.jpg', temple)

if __name__ == '__main__':
    
    # 左右物料定位
    # js_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate'
    # train_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\train_dir'
    # test_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\test_tune\腐蚀点模型测试数据\left'
    # roi_vis_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\test_tune\vis_roi'
    # img_roi = main_fun(js_dir, test_path, train_dir=train_dir, flag='left', roi_vis_path=roi_vis_path)


    js_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\apple'
    train_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\apple'
    test_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\apple_test'
    roi_vis_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\test_tune\vis_roi'
    zero_mask = main_fun(js_dir, test_path, train_dir=train_dir, flag='apple_logo', roi_vis_path=None)
    
    
    
