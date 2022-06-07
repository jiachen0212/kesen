# coding=utf-8
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from debug import help
import shutil 
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import random

def label_check(js_path, defect):
    try:
        data = json.load(open(js_path, 'r'))
    except:
        return 0

    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            def_ = cls_['label']
            # def_ = cls_['labels'][0]
            if def_ == defect:
                return 1
    return 0


def json_label_check(js_path, defects, defcet_nums):
    try:
        data = json.load(open(js_path, 'r'))
    except:
        return 0

    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            def_ = cls_['label']
            defcet_nums[defects.index(def_)] += 1
    return 1


def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def split_left_right(path,left_dir, right_dir, guang=None):
    mkdir(left_dir)
    mkdir(right_dir)

    flag_dict = {'suidao': ['2', '1'], 'tongzhou': ['1', '2']}

    files = [a for a in os.listdir(path) if '.bmp' in a]
    for file in files:
        prefix, postfix = os.path.splitext(file)[:2]
        js_file = prefix +'.json'
        split_name = file.split('-')
        assert len(split_name) == 3
        if split_name[1] == flag_dict[guang][0]:
            org1, std1 = os.path.join(path, file), os.path.join(right_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(right_dir, js_file)
        elif split_name[1] == flag_dict[guang][1]:
            org1, std1 = os.path.join(path, file), os.path.join(left_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(left_dir, js_file)
        else:
            print('data bug...')
        
        shutil.copy(org1, std1)
        if os.path.exists(org2):
            shutil.copy(org2, std2)

    print('split left and right done.~')


def roi_cut_imgtest(dir_, roi, split_target, cuted_dir):
    imgs = [a for a in os.listdir(dir_) if '.bmp' in a ]
    for name in imgs:
        img_path = os.path.join(dir_, name)
        img = Image.open(img_path)
        img = np.asarray(img)
        img_roied = img[roi[1]:roi[3], roi[0]:roi[2]]
        h, w = img_roied.shape[:2]
        sub_h, sub_w = h//split_target[1], w//split_target[0]
        for i in range(split_target[0]):
            for j in range(split_target[1]):
                sub_img = img_roied[sub_h*j: sub_h*(j+1), sub_w*i: sub_w*(i+1)]
                sub_name = name.split('.')[0]+'_{}_{}.jpg'.format(j,i)
                cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img)


def defect_random2_ignore(cur_defect, js_dir):
    js_paths = [os.path.join(js_dir, a) for a in os.listdir(js_dir) if '.json' in a]
    for js_path in js_paths:
        try:
            data = json.load(open(js_path, 'r'))
        except:
            print('bad json')
            continue 
        data1 = data.copy()
        data1['shapes'] = []
        if len(data['shapes']) > 0:
            for cls_ in data['shapes']:
                cls_1 = cls_.copy()
                def_ = cls_['label']
                if def_ == cur_defect:
                    seed = random.random()
                    if seed > 0.2:
                        cls_1['label'] = "ignore"
                        assert len(cls_1['labels']) == 1
                        cls_1['labels'] = ["ignore"]
                        data1['shapes'].append(cls_1)
                    else:
                        data1['shapes'].append(cls_)
                else:
                    # 除cur_defect之外的其他缺陷
                    data1['shapes'].append(cls_)
        data1_ = json.dumps(data1, indent=4)
        with open(js_path, 'w') as js_file:
            js_file.write(data1_)


def remove_dir(dir_):
    fs = [os.path.join(dir_, a) for a in os.listdir(dir_)]
    for f in fs:
        os.remove(f)
    os.rmdir(dir_)


def merge(save_dir, roi, data_dir):
    full_img = np.zeros((4885*4, 3628*2, 3))
    full_ = np.zeros((22000, 8192, 3))
    for i in range(2):
        for j in range(4):
            path = os.path.join(save_dir, '1-1-5_{}_{}_.bmp'.format(i, j))
            img = cv2.imread(path)  # 竖直_水平
            h,w = img.shape[:2]
            full_img[h*j:h*(j+1), w*i:w*(i+1)] = img
    
    full_[roi[1]:roi[3], roi[0]:roi[2],:] = full_img
    cv2.imwrite(os.path.join(save_dir, '1-1-5.jpg'), full_)

    return full_


if __name__ == '__main__':

    # flag1 split left right, then cut sub_bins. [suidao   guang]
    # flag2 calculate defect nums
    # flag3 split train and test 
    # flag4 train.txt test.txt defect nums
    # flag5 slim fushidian defect

    guang_type = 'tongzhou'   # 'suidao', fsmc, fs  
    flag = 1

    if guang_type in ['suidao']:
        rois = [(1200, 2026, 8192, 21650), (0, 2100, 7256, 21640)]
        defects = ["heixian", "fushidian", "huichen", "zangwu", "ignore", "znagwu"]
        split_target = (2, 4) 
    elif guang_type in ['tongzhou']:
        rois = [(1200, 600, 8192, 21000), (0, 500, 6800, 20800)]
        defects = ["disuanyise-dm", "ignore", "dds-dm-pengshang", "dds-dm-huashang", "liangyin-dm"]
        split_target = (2, 4)
    elif guang_type in ['fsmc']:
        defects = ["zangwuyise",  "ignore"]
        roi = (1500, 300, 15000, 20600)
        split_target = (4, 4)
    elif guang_type in ['fs']:
        defects = ["ignore", "dds-dm-pengshang", "dds-dm-huashang"]
        roi = (0, 700, 16000, 40800)
        split_target = (4, 8)

    defcet_nums = [0]*len(defects)
    # path需包含img和json
    # path = '/data/home/jiac
    # hen/data/seg_data/kesen/0524/data/heixian/yinbaise'
    # path = '/data/home/jiachen/data/seg_data/kesen/0601/disuanyise'
    # path = '/data/home/jiachen/data/seg_data/kesen/0601/zangwuyise/yinbaise'
    # save_dir = '/data/home/jiachen/data/seg_data/kesen/0601/fsmc/cuted_dir'
    # path = '/data/home/jiachen/data/seg_data/kesen/0601/liangyin/yinbaise'
    # path = '/data/home/jiachen/data/seg_data/kesen/0601/dds_dm/hs/yinbaise/tz'  # /data/home/jiachen/data/seg_data/kesen/0601/tz/cuted_dir/hs/yinbaise/tz
    path = '/data/home/jiachen/data/seg_data/kesen/0601/dds_dm/ps/tz' # /data/home/jiachen/data/seg_data/kesen/0601/tz/cuted_dir/dds_dm/ps/tz
    save_dir = '/data/home/jiachen/data/seg_data/kesen/0601/tz/cuted_dir'
    mkdir(save_dir)

    # 对物料在左在右进行划分
    left_dir = os.path.join(path, 'left')
    right_dir = os.path.join(path, 'right')

    if flag == 1:
        if guang_type in ["suidao", "tongzhou"]:
            split_left_right(path, left_dir, right_dir, guang=guang_type)
            names = path.split('/')[-3:]
            cuted_dir = os.path.join(save_dir, names[0], names[1], names[2])
            print(cuted_dir)
            mkdir(cuted_dir)
            help(left_dir, rois[1], split_target, cuted_dir)
            help(right_dir, rois[0], split_target, cuted_dir)
            remove_dir(left_dir)
            remove_dir(right_dir)
        else:
            names = path.split('/')[-3:]
            cuted_dir = os.path.join(save_dir, names[0], names[1], names[2])
            mkdir(cuted_dir)
            help(path, roi, split_target, cuted_dir)

    elif flag == 2:
        cuted_dir = '/data/home/jiachen/data/seg_data/kesen/0601/tz/cuted_dir/0601/liangyin/yinbaise/'
        js_paths = [os.path.join(cuted_dir, a) for a in os.listdir(cuted_dir) if '.json' in a]
        for js_path in js_paths:
            json_label_check(js_path, defects, defcet_nums)
        a = ''
        for ind, b in enumerate(defects):
            a += '{}: {}, '.format(b, defcet_nums[ind])
        print(a)

    elif flag == 3:
        train_txt = open('./{}_train.txt'.format(guang_type), 'a')
        test_txt = open('./{}_test.txt'.format(guang_type), 'a')
        heixain, no_heixian = [], []
        # 针对某一缺陷写一个train test拆分.
        # cuted_dir = '/data/home/jiachen/data/seg_data/kesen/0601/tz/cuted_dir/kesen/0601/disuanyise/'
        # cuted_dir = '/data/home/jiachen/data/seg_data/kesen/0601/fsmc/cuted_dir/0601/zangwuyise/yinbaise/'
        cuted_dir = '/data/home/jiachen/data/seg_data/kesen/0601/tz/cuted_dir/0601/liangyin/yinbaise/'
        js_paths = [os.path.join(cuted_dir, a) for a in os.listdir(cuted_dir) if '.json' in a]
        print(len(js_paths))
        for js_path in js_paths:
            if label_check(js_path, 'liangyin-dm'):  # disuanyise-dm
                heixain.append(js_path)
            else:
                no_heixian.append(js_path)
        print(len(heixain), len(no_heixian))
        random.shuffle(heixain)
        random.shuffle(no_heixian)
        # 3 7 拆分
        train_hx, train_no_hx = int(len(heixain)*0.7), int(len(no_heixian)*0.7)
        for hx in heixain[:train_hx]:
            hx = os.path.basename(hx)
            prefix, postfix = os.path.splitext(hx)[:2]
            line = '{}||{}\n'.format(cuted_dir[5:]+prefix+'.bmp', cuted_dir[5:]+hx)
            train_txt.write(line)
        
        for nohx in no_heixian[:train_no_hx]:
            nohx = os.path.basename(nohx)
            prefix, postfix = os.path.splitext(nohx)[:2]
            line = '{}||{}\n'.format(cuted_dir[5:]+prefix+'.bmp', cuted_dir[5:]+nohx)
            # print(line)
            train_txt.write(line)
        
        for hx in heixain[train_hx:]:
            hx = os.path.basename(hx)
            prefix, postfix = os.path.splitext(hx)[:2]
            line = '{}||{}\n'.format(cuted_dir[5:]+prefix+'.bmp', cuted_dir[5:]+hx)
            test_txt.write(line)
        
        for nohx in no_heixian[train_no_hx:]:
            nohx = os.path.basename(nohx)
            prefix, postfix = os.path.splitext(nohx)[:2]
            line = '{}||{}\n'.format(cuted_dir[5:]+prefix+'.bmp', cuted_dir[5:]+nohx)
            print(line)
            test_txt.write(line)

    elif flag == 4:
        trains = open('./{}_train.txt'.format(guang_type), 'r').readlines()
        tests = open('./{}_test.txt'.format(guang_type), 'r').readlines()
        js_paths = ['/data'+a[:-1].split('||')[1] for a in tests]
        for js_path in js_paths:
            json_label_check(js_path, defects, defcet_nums)
        a = ''
        for ind, b in enumerate(defects):
            a += '{}: {}, '.format(b, defcet_nums[ind])
        print('train: {}'.format(a))
    
    elif flag == 5:
        # path = /data/home/jiachen/data/seg_data/kesen/0524/data/heixian/yinbaise
        defect_random2_ignore('fushidian', path)
