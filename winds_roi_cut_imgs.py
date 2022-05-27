# coding=utf-8
import os
from PIL import Image
import numpy as np
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


def split_left_right(path,left_dir, right_dir):
    mkdir(left_dir)
    mkdir(right_dir)
    # 区分物料在左还是在右
    files = [a for a in os.listdir(path) if '.bmp' in a]
    for file in files:
        js_file = file.split('.')[0]+'.json'
        split_name = file.split('-')
        assert len(split_name) == 3
        if split_name[1] == '2':
            org1, std1 = os.path.join(path, file), os.path.join(right_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(right_dir, js_file)
        elif split_name[1] == '1':
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

    # flag1 split left right, then cut sub_bins.
    # flag2 merge img 
    # flag3 calculate defect nums

    flag = 4
    split_target = (2, 4)
    rois = [(1200, 2026, 8192, 21650), (0, 2100, 7256, 21640)] 
    defects = ["heixian", "fushidian", "huichen", "zangwu", "ignore", "znagwu"]
    defcet_nums = [0]*len(defects)

    data_list = [r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\黑线\银白色\0523']
    save_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\cuted_dir'
    print(save_dir)
    mkdir(save_dir)

    # path需要包含img和json
    for path in data_list:
        # 对物料在左在右进行划分
        left_dir = os.path.join(path, 'left')
        right_dir = os.path.join(path, 'right')
        if flag == 1:
            split_left_right(path, left_dir, right_dir)
            names = path.split('\\')[-3:]
            cuted_dir = os.path.join(save_dir, names[0], names[1], names[2])
            mkdir(cuted_dir)
            help(left_dir, rois[1], split_target, cuted_dir)
            help(right_dir, rois[0], split_target, cuted_dir)
        elif flag == 2:
            tmp_dir = r'C:\Users\15974\Desktop\tmp_dir'
            sub_img_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\cuted_dir\黑线\银白色\0523'
            merge(tmp_dir, rois[1], sub_img_dir)
        elif flag == 3:
            cuted_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\cuted_dir\黑线\银白色\0523'
            js_paths = [os.path.join(cuted_dir, a) for a in os.listdir(cuted_dir) if '.json' in a]
            for js_path in js_paths:
                json_label_check(js_path, defects, defcet_nums)
            a = ''
            for ind, b in enumerate(defects):
                a += '{}: {}, '.format(b, defcet_nums[ind])
            print(a)

        elif flag == 4:
            train_txt = open('./hx_train.txt', 'w')
            test_txt = open('./hx_test.txt', 'w')
            heixain, no_heixian = [], []
            # 针对黑线缺陷, 写一个train test拆分.
            cuted_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\cuted_dir\黑线\银白色\0523'
            js_paths = [os.path.join(cuted_dir, a) for a in os.listdir(cuted_dir) if '.json' in a]
            for js_path in js_paths:
                if label_check(js_path, 'heixian'):
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
                pre = '/home/jiachen/data/seg_data/kesen/0524/data/heixian/yinbaise/cuted_dir/data/heixian/yinbaise/'
                line = '{}||{}\n'.format(pre+hx.split('.')[0]+'.bmp', pre+hx)
                train_txt.write(line)
            
            for nohx in no_heixian[:train_no_hx]:
                nohx = os.path.basename(nohx)
                pre = '/home/jiachen/data/seg_data/kesen/0524/data/heixian/yinbaise/cuted_dir/data/heixian/yinbaise/'
                line = '{}||{}\n'.format(pre+nohx.split('.')[0]+'.bmp', pre+nohx)
                print(line)
                train_txt.write(line)
            
            for hx in heixain[train_hx:]:
                hx = os.path.basename(hx)
                pre = '/home/jiachen/data/seg_data/kesen/0524/data/heixian/yinbaise/cuted_dir/data/heixian/yinbaise/'
                line = '{}||{}\n'.format(pre+hx.split('.')[0]+'.bmp', pre+hx)
                test_txt.write(line)
            
            for nohx in no_heixian[train_no_hx:]:
                nohx = os.path.basename(nohx)
                pre = '/home/jiachen/data/seg_data/kesen/0524/data/heixian/yinbaise/cuted_dir/data/heixian/yinbaise/'
                line = '{}||{}\n'.format(pre+nohx.split('.')[0]+'.bmp', pre+nohx)
                print(line)
                test_txt.write(line)







