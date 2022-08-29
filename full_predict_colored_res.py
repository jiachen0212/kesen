# coding=utf-8
'''
suidao-sdk 
'''
import warnings

from pyrsistent import v
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


def localize_one_edge(source_image, find_in_vertical=True, thre=None, expend=200):
    # timestamp_start = time.perf_counter()
    if len(source_image.shape) == 2:
        source_image = source_image[:, :, None]

    h, w = source_image.shape[:2]
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

    return up_bound, low_bound


def mkdir(res_dir):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


def sdk_pre(img_t, mean_, std_):
    img_t = img_t[np.newaxis,:,:,:]
    img = np.array(img_t, dtype=np.float32)
    img -= np.float32(mean_)
    img /= np.float32(std_)
    img = np.transpose(img, [0, 3, 1, 2])
    return img



def cv_imread_by_np(filePath, clr_type=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), clr_type)

    return cv_img



def heixianABC(defect_values, A, B, lengthB, distanceB, B_num, boxes):
    heixian_len = len(defect_values)
    pop_ind = []
    ABvalue_indexs = [ind for ind in range(heixian_len) if (defect_values[ind] <= B and defect_values[ind] >= A)]
    # 这些ABvalue_lower_lengthB要做距离聚类, 聚类后条数<B_num就舍弃不检出
    ABvalue_lower_lengthB = [ind for ind in ABvalue_indexs if boxes[ind][3]-boxes[ind][1] <= lengthB]
    for i in range(len(ABvalue_lower_lengthB)):
        bins = []
        for j in range(i, len(ABvalue_lower_lengthB)):
            center1 = [(boxes[ABvalue_lower_lengthB[i]][0]+boxes[ABvalue_lower_lengthB[i]][2])/2, (boxes[ABvalue_lower_lengthB[i]][1]+boxes[ABvalue_lower_lengthB[i]][3])/2]
            center2 = [(boxes[ABvalue_lower_lengthB[j]][0]+boxes[ABvalue_lower_lengthB[j]][2])/2, (boxes[ABvalue_lower_lengthB[j]][1]+boxes[ABvalue_lower_lengthB[j]][3])/2]
            # 中心点abs(x)<=10,认为heixain在一条直线上 
            if abs(center1[0] - center2[0]) <= 10 and abs(center1[1] - center2[1]) <= distanceB:
                bins.append(ABvalue_lower_lengthB[j])
        if len(bins) <= B_num:
            pop_ind.extend(bins)
    pop_ind = list(set(pop_ind))

    return pop_ind


def diangui_help(defect_nums, area_large, area2_med, area_small, num_large, num_small, dis50, boxes, areas):
    # 1.面积小于area_small的都可直接放过
    pop_ind = [ind for ind in range(defect_nums) if areas[ind] <= area_small]
    # 2. 面积在[area_s, area_l]间做聚类, 数量<=num_放掉
    for k in range(2):
        if k == 0:
            area_l, arae_s, num_ = area2_med, area_small, num_large
            area_between = [ind for ind in range(defect_nums) if (areas[ind] <= area_l and areas[ind] > arae_s)]
        else:
            area_l, arae_s, num_ = area_large, area2_med, num_small
            area_between = [ind for ind in range(defect_nums) if (areas[ind] < area_l and areas[ind] > arae_s)]
        if len(area_between) <= num_:
            pop_ind.extend(area_between)
        for i in range(len(area_between)):
            bins = []
            for j in range(i, len(area_between)):
                p1, p2  = boxes[area_between[i]], boxes[area_between[j]]
                distance_i_j = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if distance_i_j <= dis50:
                    bins.append(area_between[j])
            if len(bins) <= num_:
                pop_ind.extend(bins)
    pop_ind = list(set(pop_ind))

    return pop_ind


def diangui(diangui_area_distance, boxes, areas, mask_sum, img_mask):
    # diangui_area_distance: [0.3/scalew*h, 0.2/scalew*h, 0.1/scalew*h, 0.08/scalew*h, 0.05/scalew*h, 50/scalew*h, 3, 1, 1]
    defect_nums = len(boxes)
    if not mask_sum:
        # 不存在AA面区域, 3个面积阈值直接取前三个
        area_large, area_med, area_small, dis50 = diangui_area_distance[:3] + [diangui_area_distance[5]]
        num_large, num_small = diangui_area_distance[5:7]
        pop_ind = diangui_help(defect_nums, area_large, area_med, area_small, num_large, num_small, dis50, boxes, areas)

        return pop_ind
    else:
        # 子图内存在AA面区域, 则AA, A的点规规则均需考虑
        area_0_3, area_0_2, area_0_1, area_0_0_8, area_0_0_5, dis50 = diangui_area_distance[:6]
        numa3, numa1, numaa1 = diangui_area_distance[6:]
        # 1. 先用A面的标准把过杀滤除, 
        pop_ind_temp = diangui_help(defect_nums, area_0_3, area_0_2, area_0_1, numa3, numa1, dis50, boxes, areas)
        # 2. 再从pop_ind_temp里找到AA面内(box的中心点和img_mask比较)的box, 用AA规则过一遍这些box看是否需要被检出.
        # 注意, pop_ind_temp均满足A规则下被放过, 故当ind_temp被AA的规则剔除, A规则的聚类bins个数变少, bins内的元素更是需要被剔除的, 不冲突.
        if len(pop_ind_temp):
            AA_inds = []
            for ind_temp in pop_ind_temp:
                box_ = boxes[ind_temp]
                center_ = (box_[0]+box_[2])//2, (box_[1]+box_[3])//2
                # 注意center_[1][0]
                if img_mask[center_[1]][center_[0]][0]:  # [0 0 0] or [255 255 255]
                    AA_inds.append(ind_temp)
            if len(AA_inds):
                # 以下计算AA规则下, 需要被过滤掉的所有aa_pop_inds
                aa_pop_inds = [ind for ind in AA_inds if areas[ind] <= area_0_0_5]
                # 2. 面积在[area_s, area_l]间做聚类, 数量<=num_放掉
                for k in range(2):
                    if k == 0:
                        area_l, arae_s, num_ = area_0_0_8, area_0_0_5, numaa1
                        area_between = [ind for ind in AA_inds if (areas[ind] <= area_l and areas[ind] > arae_s)]
                    else:
                        area_l, arae_s, num_ = area_0_1, area_0_0_8, numaa1
                        area_between = [ind for ind in AA_inds if (areas[ind] < area_l and areas[ind] > arae_s)]
                    if len(area_between) <= num_:
                        aa_pop_inds.extend(area_between)
                    for i in range(len(area_between)):
                        bins = []
                        for j in range(i, len(area_between)):
                            p1, p2  = boxes[area_between[i]], boxes[area_between[j]]
                            distance_i_j = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                            if distance_i_j <= dis50:
                                bins.append(area_between[j])
                        if len(bins) <= num_:
                            aa_pop_inds.extend(bins)
                aa_pop_inds = list(set(aa_pop_inds))
                if len(aa_pop_inds):
                    for tmp in AA_inds:
                        if tmp not in aa_pop_inds:
                            # tmp需要被检出
                            pop_ind_temp.remove(tmp)

        return pop_ind_temp  




def roi_and_cut_subs(img_path, roi, split_target, cuted_dir):
    _, basename = os.path.dirname(img_path), os.path.basename(img_path)
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


def label2colormap(map_):
    m = map_.astype(np.uint8)
    r, c = m.shape[:2]
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2
    cmap[:, :, 2] = (m & 4) << 5

    return cmap
    

def sdk_post(heixian_index, diangui_index, diangui_area_distance, predict_index, prdect_score_map, defects_lists, test_image, Confidence=None, num_thres=None, apple_logo_mask=None):
    pass





if __name__ == "__main__":

    heixian_index = 2

    # diangui适用的defects
    diangui_defects = ['huashang', 'zangwu', 'heidian', 'fushidian', 'zhenkong', 'madian', 
    'aokeng', 'kailie', 'keli', 'fenchen', 'maoxian', 'xianwei', 'suoshui', 'baidian', 'lianghen']
    diangui_area_distance = [(np.sqrt(0.2)/0.025)**2, (np.sqrt(0.1)/0.025)**2, (np.sqrt(0.08)/0.025)**2, (np.sqrt(0.05)/0.025)**2, (np.sqrt(0.02)/0.025)**2, 50/0.025]
    apple_logo_mask = False
    
    guang_type = 'suidao'
    onnx_name = 'station3_20220626_suidao_2000iter.onnx'

    # heixian A B C 三等级灰度值 
    A, B, C = 77, 107, 210
    lengthB, lengthC = 5/0.025, 50/0.025
    distanceB, distanceC = 10/0.025, 35/0.025
    B_num, C_num = 6, 3
    # [[77, 107, 210], [5, 50], [10, 35],[6, 3]]
    heixianban = [[A, B, C], [lengthB, lengthC], [distanceB, distanceC],[B_num, C_num]]

    # model mean && std
    mean_ = [123.675, 116.28, 103.53]
    std_ = [58.395, 57.12, 57.375]

    root_path = r'D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\sdk_test'

    if guang_type == 'suidao':
        defects = ['bg', 'fushidian', 'heixian', 'zangwu']
    split_target = (2, 4)

    # 1. 命中diangui的defect-index
    diangui_index = []
    for def_ in diangui_defects:
        try: 
            diangui_index.append(defects.index(def_))
        except:
            continue
    # 2.置信度分数[list]: 可针对各个子缺陷设置不同的置信度阈值
    Confidence = [0.5] * len(defects)
    # 3.面积过滤阈值[list]: 像素个数小于num_thres=10的不检出, 可针对各个子缺陷设置不同的面积阈值
    num_thres = [32] * len(defects)
    # 4. 点状缺陷至少满足面积>=0.02/0.025.  
    for iid in diangui_index:
        num_thres[iid] = math.ceil((np.sqrt(0.02)/0.025)**2)
     
    test_dir = os.path.join(root_path, guang_type, 'test_dir')
    test_img = os.path.join(test_dir, 'test.bmp')
    res_dir = os.path.join(root_path, guang_type, 'res_dir')
    mkdir(res_dir)

    # model deploy size 
    size = [2000, 3000]
    # load onnx
    onnx_path = os.path.join(root_path, guang_type, onnx_name)
    onnx_session = ort.InferenceSession(onnx_path)

    name = os.path.basename(test_img)
    full_img = cv_imread_by_np(test_img)  # cv2.imdecode: RGB 
    W_full, H_full = full_img.shape[:2]  # 22000, 8192 
    # locate 
    a, b = localize_one_edge(full_img, find_in_vertical=True, thre=None, expend=200)
    c, d = localize_one_edge(full_img, find_in_vertical=False, thre=None, expend=200)
    roi = [c, a, d, b]
    img = full_img
    cuted_dir = os.path.join(test_dir, 'right')
    cuted_infer_dir = os.path.join(test_dir, 'right_res')
    mkdir(cuted_dir)
    mkdir(cuted_infer_dir)
    # 切割子图
    h_, w_ = roi_and_cut_subs(test_img, roi, split_target, cuted_dir)  # 4k, 3k
    
    scale_h, scale_w = h_ / size[1], w_ / size[0] 
    num_thres = [a / (scale_h*scale_w) for a in num_thres]

    # 点规distance: box的左上角点distance.
    diangui_area_distance = [a / (scale_h*scale_w) for a in diangui_area_distance]
    # [0.02,0.08] dis>50mm都放过, [0.08,0.1] dis<=50, <=3放过, [0.1,0.2] dis<=50, <=1放过
    diangui_area_distance += [3, 1]
     
    full_index_predict = np.zeros((W_full, H_full))
    full_score_predict = np.zeros((W_full, H_full))
    for i in range(split_target[0]):
        for j in range(split_target[1]):
            Name = name.split('.')[0]+'_{}_{}.bmp'.format(j,i)
            img_name = os.path.join(cuted_dir, Name)
            img_base = Image.open(img_name) 
            img_base = np.asarray(img_base)
            img = cv2.resize(img_base, (size[0], size[1]))
            img_ = sdk_pre(img, mean_, std_)
            onnx_inputs = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
            onnx_predict = onnx_session.run(None, onnx_inputs)
            predict_index_map = np.argmax(onnx_predict[0], axis=1)
            predict = softmax(onnx_predict[0], 1)
            score_map = np.max(predict[0, :, :, :], axis=0)
            predict_index_map = predict_index_map[0, :, :]   
            # cv2.INTER_NEAREST, 最近邻插值
            org_predict_index_map = cv2.resize(predict_index_map, (w_, h_), interpolation=cv2.INTER_NEAREST)
            org_predict_score_map = cv2.resize(score_map, (w_, h_), interpolation=cv2.INTER_NEAREST)
            full_index_predict[a+j*h_:a+(j+1)*h_, c+i*w_:c+(i+1)*w_] = org_predict_index_map
            full_score_predict[a+j*h_:a+(j+1)*h_, c+i*w_:c+(i+1)*w_] = org_predict_score_map
    full_index_predict = full_index_predict.astype(np.uint8)
    # full_index_predict && full_score_predict, 做点规,黑线板 等后处理
    # sdk_post(heixian_index, diangui_index, diangui_area_distance, full_index_predict, full_score_predict, defects, full_img, Confidence=None, num_thres=None, apple_logo_mask=None):
    

    # heixian
    mask = np.array(full_index_predict == 2, np.uint8)
    temo_predict = np.zeros(full_index_predict.shape)
    cc_output = cv2.connectedComponentsWithStats(mask, 8)
    num_contours  = cc_output[0]   
    cc_stats = cc_output[2]   
    cc_labels = cc_output[1]   

    for label in range(1, num_contours):
        x = cc_stats[label, cv2.CC_STAT_LEFT]
        y = cc_stats[label, cv2.CC_STAT_TOP]
        w = cc_stats[label, cv2.CC_STAT_WIDTH]
        h = cc_stats[label, cv2.CC_STAT_HEIGHT]
        area = cc_stats[label, cv2.CC_STAT_AREA]
        temp = np.array(cc_labels == label, np.uint8)
        score_temp = temp * full_score_predict
        mean_score = np.sum(score_temp) / area
        
        if (area >= 32) and (mean_score >= 0.5):
            defect_roi = full_img[y:y+h, x:x+w] 
            h1, w1, c = defect_roi.shape[:3]
            hx_value = np.sum(defect_roi) / (h1*w1*c)
            temo_predict += temp * label 
    
    colored_predict = label2colormap(temo_predict)
    # cv2.imwrite('./hx.jpg', colored_predict)


    
    
                
            
