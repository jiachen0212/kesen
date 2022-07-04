# coding=utf-8

'''
tongzhou or suidao 的sdk脚本
'''

from turtle import distance
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
# from misc.contour_resize import resize_contour

def mkdir(res_dir):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


def label2colormap(map_):
    m = map_.astype(np.uint8)
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



def heixianABC(heixian_len, defect_values, heixian_A, heixian_B, lengthB, distanceB, heixian_B_num, boxes):
    pop_ind = []
    ABvalue_indexs = [ind for ind in range(heixian_len) if (defect_values[ind] <= heixian_B and defect_values[ind] >= heixian_A)]
    # 这些ABvalue_lower_lengthB要做距离聚类, 聚类后条数<heixian_B_num就舍弃不检出
    ABvalue_lower_lengthB = [ind for ind in ABvalue_indexs if boxes[ind][3]-boxes[ind][1] <= lengthB]

    for i in range(len(ABvalue_lower_lengthB)):
        bins = []
        for j in range(i, ABvalue_lower_lengthB):
            center1 = [(boxes[ABvalue_lower_lengthB[i]][0]+boxes[ABvalue_lower_lengthB[i]][2])/2, (boxes[ABvalue_lower_lengthB[i]][1]+boxes[ABvalue_lower_lengthB[i]][3])/2]
            center2 = [(boxes[ABvalue_lower_lengthB[j]][0]+boxes[ABvalue_lower_lengthB[j]][2])/2, (boxes[ABvalue_lower_lengthB[j]][1]+boxes[ABvalue_lower_lengthB[j]][3])/2]
            # 中心点abs(x)<=10,认为heixain在一条直线上 
            if abs(center1[0] - center2[0]) <= 10 and abs(center1[1] - center2[1]) <= distanceB:
                bins.append(ABvalue_lower_lengthB[j])
        if len(bins) < heixian_B_num:
            pop_ind.extend(bins)
    pop_ind = list(set(pop_ind))

    return pop_ind


def sdk_post(input_img, defect_list, onnx_predict, predict, heixianban, scale_h, Confidence=None, num_thres=None):
    predict_result = dict()  
    num_class = predict.shape[1]
    map_ = np.argmax(onnx_predict[0], axis=1)
    mask_ = map_[0, :, :]
    temo_predict = np.zeros(mask_.shape)
    mask_map = np.max(predict[0, :, :, :], axis=0)

    for cls in range(1, num_class):
        # {fushidian: [[score], [box], [area], [defect_value]], heixian: [[score], [box], [area], [defect_value]].. }
        if defect_list[cls] not in predict_result:
            predict_result[defect_list[cls]] = [[],[],[],[]]   

        mask = np.array(mask_ == cls, np.uint8)
        # 使用connectedComponentsWithStats能够直接输出面积和boundingbox
        cc_output = cv2.connectedComponentsWithStats(mask, 8)
        num_contours  = cc_output[0]  # 连通域的个数
        cc_stats = cc_output[2]   # 各个连通域的(x, y, width, height, area) 
        cc_labels = cc_output[1]  # 整张图的预测label结果
        
        # for each contour
        for label in range(1, num_contours):
            x = cc_stats[label, cv2.CC_STAT_LEFT]
            y = cc_stats[label, cv2.CC_STAT_TOP]
            w = cc_stats[label, cv2.CC_STAT_WIDTH]
            h = cc_stats[label, cv2.CC_STAT_HEIGHT]
            area = cc_stats[label, cv2.CC_STAT_AREA]
            temp = np.array(cc_labels == label, np.uint8)
            score_temp = temp * mask_map
            mean_score = np.sum(score_temp) / area
            
            if (area >= num_thres[cls]) and (mean_score >= Confidence[cls]):
                if cls == 2:
                    # heixian,计算原图中缺陷的灰度值.
                    defect_roi = input_img[y:y+h, x:x+w] 
                    # cv2.imwrite('./heixian_{}.jpg'.format(label), defect_roi)
                    h1, w1, c = defect_roi.shape[:3]
                    hx_value = np.sum(defect_roi) / (h1*w1*c)
                    predict_result[defect_list[cls]][3].append(hx_value)
                else:
                    temo_predict += temp * label 
                    # 非heixian,暂不计算原图中缺陷的灰度值
                    predict_result[defect_list[cls]][3].append(0)
                predict_result[defect_list[cls]][0].append(mean_score)
                predict_result[defect_list[cls]][1].append([x, y, x+w, y+h])
                predict_result[defect_list[cls]][2].append(area)

        # 根据黑线板指标, 定义优化一些heixian的过杀.
        if cls == 2:
            scores = predict_result[defect_list[cls]][0]
            boxes = predict_result[defect_list[cls]][1]
            areas = predict_result[defect_list[cls]][2]
            defect_values = predict_result[defect_list[cls]][3]
            heixian_len = len(defect_values)

            # 客户给到定制信息
            heixian_A, heixian_B, heixian_C = heixianban[0][:3]   
            lengthB, lengthC = heixianban[1][0] / scale_h, heixianban[1][1] / scale_h
            distanceB, distanceC = heixianban[2][0] / scale_h, heixianban[2][1] / scale_h
            heixian_B_num, heixian_C_num = heixianban[3][:2]
            
            pop_b = heixianABC(heixian_len, defect_values, heixian_A, heixian_B, lengthB, distanceB, heixian_B_num, boxes)
            pop_c = heixianABC(heixian_len, defect_values, heixian_B, heixian_C, lengthC, distanceC, heixian_C_num, boxes)
            all_pop = list(set(pop_b+pop_c)) 
            for ind_ in all_pop:
                boxes.pop(ind_)
                scores.pop(ind_)
                areas.pop(ind_)
            predict_result[defect_list[cls]] = [scores, boxes, areas]

    return temo_predict, predict_result


def roi_cut_imgtest(img_path, roi, split_target, cuted_dir):
    basename = os.path.basename(img_path)
    name = basename.split('.')[0]
    img = Image.open(img_path)
    img = np.asarray(img)
    img_roied = img[roi[1]:roi[3], roi[0]:roi[2]]
    h, w = img_roied.shape[:2]
    sub_h, sub_w = h//split_target[1], w//split_target[0]
    for i in range(split_target[0]):
        for j in range(split_target[1]):
            sub_img = img_roied[sub_h*j: sub_h*(j+1), sub_w*i: sub_w*(i+1)]
            sub_name = name.split('.')[0]+'_{}_{}.bmp'.format(j, i)
            sub_img_bgr = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img_bgr)

    return sub_h, sub_w


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
    
    guang_type = 'suidao'
    onnx_name = 'station3_20220626_suidao_2000iter.onnx'

    # 8K 0.027mm/pixel suidao光工位的像素分辨率, 后面会和黑线板的检出长度mm结合计算. 
    # heixian缺陷, ABC三个灰度值等级
    heixian_A, heixian_B, heixian_C = 105, 109, 110
    lengthB, lengthC = 5*0.027, 50*0.027
    distanceB, distanceC = 10*0.027, 35*0.027
    heixian_B_num, heixian_C_num = 6, 3
    heixianban = [[heixian_A, heixian_B, heixian_C], [lengthB, lengthC], [distanceB, distanceC],[heixian_B_num, heixian_C_num]]

    # 模型的mean和std
    mean_ = [123.675, 116.28, 103.53]
    std_ = [58.395, 57.12, 57.375]

    root_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test'

    # roi边界冗余,纵横起始点; left用rois[1], right用rois[0] 
    if guang_type == 'suidao':
        rois = [(1200, 2026, 8192, 21650), (0, 2100, 7256, 21640)] 
        defects = ['bg', 'fushidian', 'heixian', 'zangwu']
    split_target = (2, 4)

    # 1. 和defcet_dict的keys一一对应
    ng_nums = [0] * len(defects)
    # 2.置信度分数[list]: 可针对各个子缺陷设置不同的置信度阈值
    Confidence = [0.5] * len(defects)
    # 3.面积过滤阈值[list]: 像素个数小于num_thres的不检出, 可针对各个子缺陷设置不同的面积阈值
    num_thres = [50] * len(defects)

    # 物料left和物料right, 共测试两张.
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

    for left_or_right_img in test_paths:
        full_img = Image.open(left_or_right_img)
        H_full, W_full = full_img.size[:2]

        # img_left or img_right
        im_name = os.path.basename(left_or_right_img)
        name = im_name.split('.')[0]
        if im_name.split('-')[1] == '1':
            roi = rois[1]
            cuted_dir = os.path.join(test_dir, 'left')
            cuted_infer_dir = os.path.join(test_dir, 'left_res')
            mkdir(cuted_dir)
            mkdir(cuted_infer_dir)
        elif im_name.split('-')[1] == '2':
            roi = rois[0]
            cuted_dir = os.path.join(test_dir, 'right')
            cuted_infer_dir = os.path.join(test_dir, 'right_res')
            mkdir(cuted_dir)
            mkdir(cuted_infer_dir)
        # 落盘sub_imgs, j_i是sub_bin的索引.sub_img的检出box的坐标信息需换算至整图坐标,需要此索引信息.
        h_, w_ = roi_cut_imgtest(left_or_right_img, roi, split_target, cuted_dir)
        
        # 子图进入模型的缩放系数
        scale_h, scale_w = h_ / size[1], w_ / size[0] 
        num_thres = [a / (scale_h*scale_w) for a in num_thres]
        # inference单张子图
        for i in range(split_target[0]):
            for j in range(split_target[1]):
                Name = name.split('.')[0]+'_{}_{}.bmp'.format(j,i)
                img_name = os.path.join(cuted_dir, Name)
                img_base = Image.open(img_name) 
                img_base = np.asarray(img_base)
                # sub_img_inference, scale sub_img
                img = cv2.resize(img_base, (size[0], size[1]))
                img_ = sdk_pre(img, mean_, std_)
                onnx_inputs = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
                onnx_predict = onnx_session.run(None, onnx_inputs)
                predict = softmax(onnx_predict[0], 1)
                map_, predict_result = sdk_post(img, defects, onnx_predict, predict, heixianban, scale_h, Confidence=Confidence, num_thres=num_thres)
                scores, boxes, areas = [],[],[]
                for k, v in predict_result.items():
                    if len(v[0]):
                        scores.extend(v[0])
                        boxes.extend(v[1])
                        areas.extend(v[2])
                mask_vis = label2colormap(map_)
                # 绘制矩形框
                if len(scores):
                    for ind, box in enumerate(boxes):
                        cv2.rectangle(mask_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                        box = [box[:2], box[2:]]
                        box1 = [[int(scale_w*a[0])+w_*i+roi[0], int(scale_h*a[1])+h_*j+roi[1]] for a in box] 
                        text = '{}, '.format(np.round(scores[ind], 2))
                        text += ''.join(str(a)+',' for a in box1)
                        text += '{}'.format(int(areas[ind]*scale_w*scale_h))
                        cv2.putText(mask_vis, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
                # re_scale sub_img
                sub_inference_img = cv2.resize(img_save, (w_, h_))
                print('save sub inference result ~.')
                cv2.imwrite(os.path.join(cuted_infer_dir, Name), sub_inference_img)
        # 合并suub_img的inference_res
        full_ = merge(H_full, W_full, name, cuted_infer_dir, roi, split_target, h_, w_)
        cv2.imwrite(os.path.join(res_dir, im_name), full_)