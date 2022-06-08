# coding=utf-8

'''
fs fsmc 的sdk脚本
'''

import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from scipy import spatial
import os
from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math

def mkdir(res_dir):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


def label2colormap(label):
    m = label.astype(np.uint8)
    r, c = m.shape[:2]
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2
    cmap[:, :, 2] = (m & 4) << 5

    return cmap


def sdk_pre(img_t, mean_, std_):
    img_t = img_t[np.newaxis,:,:,:]
    img = np.array(img_t, dtype=np.float32)
    img -= np.float32(mean_)
    img /= np.float32(std_)
    img = np.transpose(img, [0, 3, 1, 2])
    return img


def check_connect_comp(img, label_index):
    mask = np.array(img == label_index, np.uint8)
    num, label = cv2.connectedComponents(mask, 8)
    return mask, num, label


def find_farthest_two_points(points, metric="euclidean"):
    """
    找出点集中最远距离的两个点：凸包 + 距离计算
    Args:
        points (numpy.ndarray, N x dim): N个d维向量
        metric ("euclidean", optional): 距离度量方式，见scipy.spatial.distance.cdist
    Returns:
        np.ndarray: 两点坐标
    """
    hull = spatial.ConvexHull(points)
    hullpoints = points[hull.vertices]
    hdist = spatial.distance.cdist(hullpoints, hullpoints, metric=metric)
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    return [hullpoints[bestpair[0]], hullpoints[bestpair[1]]]


def sdk_post(predict, defects, Confidence=None, num_thres=None):
    defects_nums = [0]*len(defects)
    boxes = []
    num_class = predict.shape[1]
    map_ = np.argmax(onnx_predict[0], axis=1)
    # print(f'pixel_classes: {np.unique(map_)}')
    mask_map = np.max(predict[0, :, :, :], axis=0)
    mask_ = map_[0, :, :]
    temo_predict = np.zeros(mask_.shape)
    for i in range(num_class):
        if i == 0:
            continue
        else:
            _, num, label = check_connect_comp(mask_, i)
            for j in range(num):
                if j == 0:
                    continue
                else:
                    temp = np.array(label == j, np.uint8)
                    score_temp = temp * mask_map
                    locate = np.where(temp > 0)
                    number_thre = len(locate[0])
                    score_j = np.sum(score_temp) / number_thre

                    if number_thre > num_thres[i] and score_j > Confidence[i]:
                        contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnt = contours[0]
                        cnt = cnt.reshape(cnt.shape[0], -1)
                        if cnt.shape[0] < 3:
                            continue

                        # 得到缺陷的外接正矩形: x, y, h, w
                        rect = cv2.boundingRect(cnt)
                        rect = [a for a in rect]
                        box_d = rect[:2] + [rect[0]+rect[2], rect[1]+rect[3]] 

                        # 得到缺陷的最小外接矩形
                        # rect = cv2.minAreaRect(cnt)
                        # # 得到旋转矩形的端点
                        # box = cv2.boxPoints(rect)
                        # box_d = np.int0(box)
                        
                        # 统计缺陷个数
                        defects_nums[i] += 1
                        boxes.append(box_d)
                        temo_predict += temp * i
                       

    return temo_predict, boxes, defects_nums


def roi_cut_imgtest(guang_type, img_path, roi, split_target, cuted_dir):
    basename = os.path.basename(img_path)
    name = basename.split('.')[0]
    img = Image.open(img_path)
    img = np.asarray(img)
    # fsmc保存的图ndim==2
    # if guang_type == 'fsmc':
    img = cv2.merge([img, img, img])
    img_roied = img[roi[1]:roi[3], roi[0]:roi[2]]
    h, w = img_roied.shape[:2]
    sub_h, sub_w = h//split_target[1], w//split_target[0]
    for i in range(split_target[0]):
        for j in range(split_target[1]):
            sub_img = img_roied[sub_h*j: sub_h*(j+1), sub_w*i: sub_w*(i+1)]
            sub_name = name.split('.')[0]+'_{}_{}.bmp'.format(j,i)
            cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img)


def merge(H_full, W_full, name, sub_imgs_dir, roi, split_target, h_, w_):
    full_img = np.zeros((h_*split_target[1], w_*split_target[0], 3))
    full_ = np.zeros((W_full, H_full, 3))
    for i in range(split_target[0]):
        for j in range(split_target[1]):
            path = os.path.join(sub_imgs_dir, '{}_{}_{}.bmp'.format(name, j, i))
            img = cv2.imread(path)  # 竖直_水平
            full_img[h_*j:h_*(j+1), w_*i:w_*(i+1)] = img
    full_[roi[1]:roi[3], roi[0]:roi[2],:] = full_img

    return full_


if __name__ == "__main__":
    
    # 在这里选择: 'fs' or 'fsmc'
    guang_type = 'fs'
    onnx_name = '800.onnx'

    # 模型的mean和std
    mean_ = [123.675, 116.28, 103.53]
    std_ = [58.395, 57.12, 57.375]

    root_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test'

    # roi边界冗余 
    if guang_type in ['fsmc']:
        defects = ["bg", "zangwuyise"]
        roi = (1500, 300, 15000, 20600)
        split_target = (4, 4)
    elif guang_type in ['fs']:
        defects = ["bg", "dds-dm-pengshang", "dds-dm-huashang"]
        roi = (0, 700, 16000, 40800)
        split_target = (4, 8)

    # 1. 和defcet_dict的keys一一对应
    ng_nums = [0] * len(defects)
    # 2.置信度分数[list]: 可针对各个子缺陷设置不同的置信度阈值
    Confidence = [0.5] * len(defects)
    # 3.面积过滤阈值[list]: 像素个数小于num_thres的不检出, 可针对各个子缺陷设置不同的面积阈值
    num_thres = [50] * len(defects)

    test_dir = os.path.join(root_path, guang_type, 'test_dir')
    test_paths = [os.path.join(test_dir, a) for a in os.listdir(test_dir) if '.bmp' in a]
    # 保存测试图像的结果
    res_dir = os.path.join(root_path, guang_type, 'res_dir')
    mkdir(res_dir)

    # 部署模型的输入尺寸
    size = [2000, 3000]
    # 导入onnx
    onnx_path = os.path.join(root_path, guang_type, onnx_name)
    onnx_session = ort.InferenceSession(onnx_path)

    for test_im in test_paths:
        im_name = os.path.basename(test_im)
        name = im_name.split('.')[0]
        full_img = Image.open(test_im)
        # full_img_np = np.asarray(full_img)
        # print(full_img_np.ndim)
        H_full, W_full = full_img.size[:2]  
        # fs 和 fsmc 图像没有物料在左或右的区别
        cuted_dir = os.path.join(test_dir, 'sub')
        cuted_infer_dir = os.path.join(test_dir, 'sub_res')
        mkdir(cuted_dir)
        mkdir(cuted_infer_dir)
        # 落盘sub_imgs, j_i是sub_bin的索引.sub_img的检出box的坐标信息需换算至整图坐标,需要此索引信息.
        roi_cut_imgtest(guang_type, test_im, roi, split_target, cuted_dir)

        # inference单张子图
        for i in range(split_target[0]):
            for j in range(split_target[1]):
                Name = name.split('.')[0]+'_{}_{}.bmp'.format(j,i)
                img_name = os.path.join(cuted_dir, Name)
                img_base = Image.open(img_name) 
                img_base = np.asarray(img_base)
                h_, w_ = img_base.shape[:2]
                scale_h, scale_w = h_ / size[1], w_ / size[0]
                # sub_img_inference, scale sub_img
                img = cv2.resize(img_base, (size[0], size[1]))
                img_ = sdk_pre(img, mean_, std_)
                onnx_inputs = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
                onnx_predict = onnx_session.run(None, onnx_inputs)
                predict = softmax(onnx_predict[0], 1)
                map_, boxes, defects_nums = sdk_post(predict, defects, Confidence=Confidence, num_thres=num_thres)
                mask_vis = label2colormap(map_)
                # 绘制矩形框
                if boxes:
                    for box in boxes:
                        cv2.rectangle(mask_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                        box = [box[:2], box[2:]]
                        # roi[1]+h_*j, roi[0]+w_*i叠加到sub_img的坐标上, 映射回整图坐标值.
                        # 并且输入模型inference的尺寸虽小了, 需要scale_h,w乘回来.
                        box1 = [[int(scale_w*a[0])+w_*i+roi[0], int(scale_h*a[1])+h_*j+roi[1]] for a in box] 
                        cv2.putText(mask_vis, ''.join(str(a)+',' for a in box1), box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
                # img_save = cv2.cvtColor(img_save, cv2.COLOR_BGR2GRAY)
                # re_scale sub_img
                sub_inference_img = cv2.resize(img_save, (w_, h_))
                print('save sub inference result ~.')
                cv2.imwrite(os.path.join(cuted_infer_dir, Name), sub_inference_img)
        # 合并suub_img的inference_res
        full_ = merge(H_full, W_full, name, cuted_infer_dir, roi, split_target, h_, w_)
        cv2.imwrite(os.path.join(res_dir, im_name), full_)
 
