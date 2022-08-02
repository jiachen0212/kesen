# coding=utf-8
'''
suidao 的sdk脚本
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
    找出点集中最远距离的两个点: 凸包 + 距离计算
    Args:
        points (numpy.ndarray, N x dim): N个d维向量
        metric ("euclidean", optional): 距离度量方式, 见scipy.spatial.distance.cdist
    Returns:
        np.ndarray: 两点坐标
    """
    hull = spatial.ConvexHull(points)
    hullpoints = points[hull.vertices]
    hdist = spatial.distance.cdist(hullpoints, hullpoints, metric=metric)
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    return [hullpoints[bestpair[0]], hullpoints[bestpair[1]]]



def heixianABC(defect_values, heixian_A, heixian_B, lengthB, distanceB, heixian_B_num, boxes):
    heixian_len = len(defect_values)
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
        if len(bins) <= heixian_B_num:
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


def sdk_post(heixianban_index, diangui_index, diangui_area_distance, input_img, img_mask, defect_list, onnx_predict, predict, heixianban, scale_h, Confidence=None, num_thres=None):
    # 计算一下img_mask的像素和:mask纯黑sum=0,则不存在AA面,就不需要出发AA的判断
    mask_sum = np.sum(img_mask)
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
                    # 非heixian,暂不计算原图中缺陷的灰度值
                    predict_result[defect_list[cls]][3].append(0)

                # dl模型的检出都做渲染, 命中黑线板, 点规等规则的则剔除box.
                temo_predict += temp * label 

                predict_result[defect_list[cls]][0].append(mean_score)
                predict_result[defect_list[cls]][1].append([x, y, x+w, y+h])
                predict_result[defect_list[cls]][2].append(area)

        # 1. 根据黑线板规则, 过滤一些heixian过杀.
        if cls == heixianban_index:
            scores = predict_result[defect_list[cls]][0]
            boxes = predict_result[defect_list[cls]][1]
            areas = predict_result[defect_list[cls]][2]
            defect_values = predict_result[defect_list[cls]][3]
            heixian_A, heixian_B, heixian_C = heixianban[0][:3]   
            lengthB, lengthC = heixianban[1][0] / scale_h, heixianban[1][1] / scale_h
            distanceB, distanceC = heixianban[2][0] / scale_h, heixianban[2][1] / scale_h
            heixian_B_num, heixian_C_num = heixianban[3][:2]
            pop_b = heixianABC(defect_values, heixian_A, heixian_B, lengthB, distanceB, heixian_B_num, boxes)
            pop_c = heixianABC(defect_values, heixian_B, heixian_C, lengthC, distanceC, heixian_C_num, boxes)
            all_pop = list(set(pop_b+pop_c)) 
            scores_, boxes_, areas_ = [], [], []
            for ind in range(len(boxes)):
                if ind not in all_pop:
                    scores_.append(scores[ind])
                    boxes_.append(boxes[ind])
                    areas_.append(areas[ind])
            predict_result[defect_list[cls]] = [scores_, boxes_, areas_]
            print('heixianban poped: {}'.format(all_pop))
        
        # 2. 根据点规规则, 过滤一些点状过杀.
        if cls in diangui_index:
            scores = predict_result[defect_list[cls]][0]
            boxes = predict_result[defect_list[cls]][1]
            areas = predict_result[defect_list[cls]][2]
            # 点规规则: diangui_area_distance： [0.3/scalew*h, 0.2/scalew*h, 0.1/scalew*h, 0.08/scalew*h, 0.05/scalew*h, 50/scalew*h, 3, 1, 1]
            all_pop = diangui(diangui_area_distance, boxes, areas, mask_sum, img_mask)
            scores_, boxes_, areas_ = [], [], []
            for ind in range(len(boxes)):
                if ind not in all_pop:
                    scores_.append(scores[ind])
                    boxes_.append(boxes[ind])
                    areas_.append(areas[ind])
            predict_result[defect_list[cls]] = [scores_, boxes_, areas_]
            print('diangui poped: {}'.format(all_pop))
        
    return temo_predict, predict_result
    

def roi_cut_imgtest(img_path, roi, split_target, cuted_dir, mask=False,logo_mask_img='', logo_roi=''):
    pre, basename = os.path.dirname(img_path), os.path.basename(img_path)
    name = basename.split('.')[0]
    img = Image.open(img_path)
    img = np.asarray(img)
    img_roied = img[roi[1]:roi[3], roi[0]:roi[2]]
    h, w = img_roied.shape[:2]
    sub_h, sub_w = h//split_target[1], w//split_target[0]
    if mask:
        mask_img = np.zeros_like(img)
        # logo区域aa面是白色, 求他全黑, zero处理即可. 
        mask_img[logo_roi[o]:logo_roi[1], roi[2]] = logo_mask_img
    for i in range(split_target[0]):
        for j in range(split_target[1]):
            sub_img = img_roied[sub_h*j: sub_h*(j+1), sub_w*i: sub_w*(i+1)]
            sub_name = name.split('.')[0]+'_{}_{}.bmp'.format(j, i)
            sub_img_bgr = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img_bgr)
            if mask:
                sub_mask_img = mask_img[sub_h*j: sub_h*(j+1), sub_w*i: sub_w*(i+1)]
                # mask的子图和原图的子图, 用后缀bmp和jpg区分
                sub_name = name.split('.')[0]+'_{}_{}.jpg'.format(j, i)
                sub_mask_img = cv2.cvtColor(sub_mask_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_mask_img)

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


def Apple_logo_locate(image_path, roi):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    # 长宽roi阈值
    image = image[roi[0]:roi[1], roi[2]:]
    otsuThe, maxValue = 0, 255  # otsuThe=46
    otsuThe, dst_Otsu = cv2.threshold(image, 1, maxValue, cv2.THRESH_OTSU)
    dst_Otsu = cv2.bitwise_not(dst_Otsu)
    # 检测出的apple-logo边上还是有点黑点, so先腐蚀(去除黑点)再膨胀(外扩白像素.)
    kernel = np.ones((50, 50), dtype=np.uint8)
    dst_Otsu1 = cv2.erode(dst_Otsu, kernel, iterations=1)
    kernel = np.ones((30, 30), dtype=np.uint8)
    dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 5)  

    # 另一套膨胀腐蚀参数
    # dst_Otsu3 = cv2.dilate(dst_Otsu1, kernel, 6) 
    # kernel_ = np.ones((40, 40), dtype=np.uint8)
    # dst_Otsu4 = cv2.dilate(dst_Otsu1, kernel_, 5)

    # 存apple-logo的mask图.
    # cv2.imwrite(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir\1.jpg', dst_Otsu2)
    # cv2.imwrite(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir\2.jpg', dst_Otsu4)

    return dst_Otsu2 



if __name__ == "__main__":
    
    # 黑线的index本应该是2的, 这里写成33,则不会触发黑线板规则.
    heixianban_index = 33

    # diangui规则适用的缺陷
    diangui_defects = ['huashang', 'zangwu', 'heidian', 'fushidian', 'zhenkong', 'madian', 'aokeng', 'kailie', 'keli', 'fenchen', 'maoxian', 'xianwei', 'suoshui', 'baidian', 'lianghen']
    # A面area阈值: 0.1,0.2, AA面area阈值: 0.05, 0.08. 单个像素按照0.025mm算整除方便. 
    diangui_area_distance = [0.3/0.025, 0.2/0.025, 0.1/0.025, 0.08/0.025, 0.05/0.025, 50/0.025]
    # 是否需要做apple-logo-mask定位
    apple_logo_mask = True
    
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

    root_path = r'D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\sdk_test'

    # roi边界冗余,纵横起始点; left用rois[1], right用rois[0] 
    if guang_type == 'suidao':
        rois = [(1200, 2026, 8192, 21650), (0, 2257, 7233, 21697)] 
        defects = ['bg', 'fushidian', 'heixian', 'zangwu']
    split_target = (2, 4)

    # 1.本隧道模型检测的缺陷, 看看命中diangui规则的index
    diangui_index = []
    for def_ in diangui_defects:
        try: 
            diangui_index.append(defects.index(def_))
        except:
            continue
    # 2.置信度分数[list]: 可针对各个子缺陷设置不同的置信度阈值
    Confidence = [0.5] * len(defects)
    # 专门卡控腐蚀点高置信度
    Confidence[1] = 0.9
    # 3.面积过滤阈值[list]: 像素个数小于num_thres=10的不检出, 可针对各个子缺陷设置不同的面积阈值
    num_thres = [10] * len(defects)
    # 4. 点状缺陷至少满足面积>=(np.sqrt(0.02)/0.025)^2. [没有cover-heixain]
    for iid in diangui_index:
        num_thres[iid] = math.ceil((np.sqrt(0.02)/0.025)**2) # 10, 32, 10, 32

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
            # 图像名称中检索到1表示本图有apple-logo, 则开启logo-mask计算
            # 除隧道工位外的模型应该不需要做这个判断, 直接调用apple-logo-mask函数即可
            logo_roi = (1185, 20000, 2500)
            logo_mask = Apple_logo_locate(left_or_right_img, logo_roi)
            roi = rois[1]
            cuted_dir = os.path.join(test_dir, 'left')
            cuted_infer_dir = os.path.join(test_dir, 'left_res')
            mkdir(cuted_dir)
            mkdir(cuted_infer_dir)
        elif im_name.split('-')[1] == '2':
            roi = rois[0]
            roi = (1200, 1000, 8192, 20480)
            cuted_dir = os.path.join(test_dir, 'right')
            cuted_infer_dir = os.path.join(test_dir, 'right_res')
            mkdir(cuted_dir)
            mkdir(cuted_infer_dir)
        # 落盘sub_imgs, j_i是sub_bin的索引.sub_img的检出box的坐标信息需换算至整图坐标,需要此索引信息.
        h_, w_ = roi_cut_imgtest(left_or_right_img, roi, split_target, cuted_dir, mask=apple_logo_mask, logo_mask_img =logo_mask, logo_roi=logo_roi)
        
        # 子图进入模型的缩放系数
        scale_h, scale_w = h_ / size[1], w_ / size[0] 
        num_thres = [a / (scale_h*scale_w) for a in num_thres]

        # 面积距离都/(scale_h*scale_w), 点规间的距离用bbox的左上角点间的距离计算.
        diangui_area_distance = [a / (scale_h*scale_w) for a in diangui_area_distance]
        # 面积[0.1,0.2]间的<=3放过, 面积大于[0.2,0.3]的<=1放过. [0.05,0.1]的<=1放过.
        diangui_area_distance += [3, 1, 1]
        

        # inference单张子图
        for i in range(split_target[0]):
            for j in range(split_target[1]):
                Name = name.split('.')[0]+'_{}_{}.bmp'.format(j,i)
                print("img: {}, ".format(Name), end="")
                img_name = os.path.join(cuted_dir, Name)
                img_base = Image.open(img_name) 
                img_base = np.asarray(img_base)
                # 读取子图对应的mask图
                img_base_mask = Image.open(img_name.split('.')[0]+'.bmp') 
                img_base_mask = np.asarray(img_base_mask)
                # sub_img_inference, scale sub_img
                img = cv2.resize(img_base, (size[0], size[1]))
                img_mask = cv2.resize(img_base_mask, (size[0], size[1]))
                img_ = sdk_pre(img, mean_, std_)
                onnx_inputs = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
                onnx_predict = onnx_session.run(None, onnx_inputs)
                predict = softmax(onnx_predict[0], 1)
                map_, predict_result = sdk_post(heixianban_index, diangui_index, diangui_area_distance, img, img_mask, defects, onnx_predict, predict, heixianban, scale_h, Confidence=Confidence, num_thres=num_thres)
                scores, boxes, areas, clsses = [],[],[],[]
                for k, v in predict_result.items():
                    if len(v[0]):
                        scores.extend(v[0])
                        boxes.extend(v[1])
                        areas.extend(v[2])
                        clsses.extend([k]*len(v[0]))
                mask_vis = label2colormap(map_)
                # 绘制矩形框
                if len(scores):
                    for ind, box in enumerate(boxes):
                        cv2.rectangle(mask_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                        box = [box[:2], box[2:]]
                        box1 = [[int(scale_w*a[0])+w_*i+roi[0], int(scale_h*a[1])+h_*j+roi[1]] for a in box] 
                        text = '{}: {}, '.format(clsses[ind], np.round(scores[ind], 2))
                        text += ''.join(str(a)+',' for a in box1)
                        text += '{}'.format(int(areas[ind]*scale_w*scale_h))
                        cv2.putText(mask_vis, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
                # re_scale sub_img
                sub_inference_img = cv2.resize(img_save, (w_, h_))
                cv2.imwrite(os.path.join(cuted_infer_dir, Name), sub_inference_img)
        # 合并suub_img的inference_res
        full_ = merge(H_full, W_full, name, cuted_infer_dir, roi, split_target, h_, w_)
        cv2.imwrite(os.path.join(res_dir, im_name), full_)
    