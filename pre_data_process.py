# -*- coding: utf-8 -*-
import cv2
import os
import shutil 
import numpy as np
import json



def cv_imread_by_np(filePath, clr_type=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), clr_type)

    return cv_img



def slim_dirs(dir_):
    im_dirs = [os.path.join(dir_, a) for a in os.listdir(dir_) if '.json' not in a]
    for im_dir in im_dirs:
        im_names = [a for a in os.listdir(im_dir) if '.bmp' in a]
        for im_name in im_names:
            shutil.copy(os.path.join(im_dir, im_name), os.path.join(dir_, im_name))  
        shutil.rmtree(im_dir)


def slim_dirs1(dir_):
    # 文件夹下还有ok ng的子名字的, 先剔除ok ng这一层. 
    dirs = [os.path.join(dir_, a) for a in os.listdir(dir_)]
    for dir_  in dirs:
        path = os.path.join(dir_, os.listdir(dir_)[0])
        ims = [a for a in os.listdir(path) if '.bmp' in a]
        print(ims)
        for im_name in ims:
            shutil.copy(os.path.join(path, im_name), os.path.join(dir_, im_name))  
            # print(os.path.join(path, im_name), os.path.join(dir_, im_name))
        shutil.rmtree(path)
    


def slim_dirs_js(dir_):
    im_dirs = [os.path.join(dir_, a) for a in os.listdir(dir_)]
    for im_dir in im_dirs:
        im_names = [a for a in os.listdir(im_dir) if '.json' in a]
        for im_name in im_names:
            shutil.copy(os.path.join(im_dir, im_name), os.path.join(dir_, im_name))  
        shutil.rmtree(im_dir)


def slim_dirs1_js(dir_):
    # 文件夹下还有ok ng的子名字的, 先剔除ok ng这一层. 
    dirs = [os.path.join(dir_, a) for a in os.listdir(dir_)]
    for dir_  in dirs:
        path = os.path.join(dir_, os.listdir(dir_)[0])
        ims = [a for a in os.listdir(path) if '.json' in a]
        for im_name in ims:
            shutil.copy(os.path.join(path, im_name), os.path.join(dir_, im_name))  
            # print(os.path.join(path, im_name), os.path.join(dir_, im_name))
        shutil.rmtree(path)


def localize_one_edge(source_image, find_in_vertical=True, thre=None, expend=200):
    
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

    return up_bound, low_bound


def generate_im_roi_json(base_dir, roi_json_name):
    ims = os.listdir(base_dir)
    name_roi = dict()
    for im in ims:
        path = os.path.join(base_dir, im)
        image = cv_imread_by_np(path)
        a, b = localize_one_edge(image, find_in_vertical=True, thre=None, expend=200)
        c, d = localize_one_edge(image, find_in_vertical=False, thre=None, expend=200)
        name_roi[im] = [int(c), int(a), int(d), int(b)]
        # check roi
        # temp = image[a:b, c:d]
        # cv2.imwrite('./1.jpg', temp)
    json_str = json.dumps(name_roi, indent=4)
    with open(roi_json_name, 'w') as json_file:
        json_file.write(json_str)


def json_label_check(js_path, defects, defcet_nums):
    try:
        data = json.load(open(js_path, 'r'))
    except:
        return 0

    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            def_ = cls_['label'].encode('gbk').decode('utf-8')
            defcet_nums[defects.index(def_)] += 1
    return 1


if __name__ == "__main__":

    # for .bmps
    # base_dir = r'D:\work\project\DL\kesen\data\腐蚀点缺陷样图\过杀\0829\0829'
    # slim_dirs1(base_dir)
    # slim_dirs(base_dir)

    # 动态生成roi
    # generate_im_roi_json(base_dir, './0829_im_roi.json')


    # for jss
    base_dir = r'D:\work\project\DL\kesen\data\20220828_JSON\20220827 工位3&4 缺陷图样\腐蚀点缺陷样图\过杀\原图'
    slim_dirs1_js(base_dir)
    slim_dirs_js(base_dir)



    # defects = [r"发黑", r"堆银", r"残银", r"破损", r"划痕", r"裂片"]
    # defcet_nums = [0]*len(defects)
    # json_dir = r'D:\work\project\beijing\Smartmore\2022\DL\vimo算法联动\0829迈维视\data_0829'
    # js_paths = [os.path.join(json_dir, a) for a in os.listdir(json_dir) if '.json' in a]
    # for js_path in js_paths:
    #     json_label_check(js_path, defects, defcet_nums)
    # a = ''
    # for ind, b in enumerate(defects):
    #     a += '{}: {}, '.format(b, defcet_nums[ind])
    # print(a)


    
