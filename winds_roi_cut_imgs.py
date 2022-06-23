# coding=utf-8
from datetime import date
import os
from torch import full_like
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from debug import help
import shutil 
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json


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


def split_left_right(path, left_dir, right_dir, guang=None):
    mkdir(left_dir)
    mkdir(right_dir)

    files = [a for a in os.listdir(path) if '.bmp' in a]
    for file in files:
        cuted_dirfix, _ = os.path.splitext(file)[:2]
        js_file = cuted_dirfix +'.json'
        
        org = os.path.join(path, file) 
        img_pil = Image.open(org)
        img = np.asarray(img_pil)
        temp1 = [a for a in img[int(img.shape[0]//2)][0]]
        temp2 = [a for a in img[int(img.shape[0]//2)][0]]

        if (np.sum(temp1) + np.sum(temp2)) < 10:
            # 'right'
            org1, std1 = os.path.join(path, file), os.path.join(right_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(right_dir, js_file)

        else:
            # 'left'
            org1, std1 = os.path.join(path, file), os.path.join(left_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(left_dir, js_file)
        
        if os.path.exists(org2):
            shutil.copy(org2, std2)
            shutil.copy(org1, std1)

    print('split left and right done.~')



def split_AC_BD(path, long_dir, short_dir, guang=None):

    mkdir(short_dir)
    mkdir(long_dir)
    files = [a for a in os.listdir(path) if '.bmp' in a]
    for file in files:
        cuted_dirfix, postfix = os.path.splitext(file)[:2]
        js_file = cuted_dirfix +'.json'

        index_b = file.split('-')[-1][0]
        assert index_b in 'ABCDabcd'
        if index_b in 'ACac':  # 长边
            org1, std1 = os.path.join(path, file), os.path.join(long_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(long_dir, js_file)
        elif index_b in 'BDbd': # 短边
            org1, std1 = os.path.join(path, file), os.path.join(short_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(short_dir, js_file)
        
        shutil.copy(org1, std1)
        if os.path.exists(org2):
            shutil.copy(org2, std2)

    print('split long and short done.~')


def merge(save_dir, roi):
    # full_img = np.zeros((4885*4, 3628*2, 3))
    # full_img = np.zeros((5075*4, 3400*2, 3))
    # full_img = np.zeros((5075*4, 3375*4, 3))
    full_img = np.zeros((5100*4, 3496*2, 3))
    full_ = np.zeros((22000, 8192, 3))
    for i in range(2):
        for j in range(4):
            path = os.path.join(save_dir, '15-1-3_{}_{}_.bmp'.format(i, j))
            img = cv2.imread(path)  # 竖直_水平
            h,w = img.shape[:2]
            full_img[h*j:h*(j+1), w*i:w*(i+1)] = img
    
    full_[roi[1]:roi[3], roi[0]:roi[2],:] = full_img
    cv2.imwrite(os.path.join(save_dir, '15-1-3.jpg'), full_)

    return full_


if __name__ == '__main__':

    # flag1 split left right, then cut sub_bins.   
    # flag2 merge img 
    # flag3 calculate defect nums

    guang_type = 'suidao'  # 'tongzhou' 'suidao', fsmc, fs; cb_suidao, cb_tongzhou
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
        defects = ["ignore", "zangwuyise"]
        roi = (1500, 300, 15000, 20600)
        split_target = (4, 4)
    elif guang_type in ['fs']:
        defects = ["ignore", "dds-dm-pengshang", "dds-dm-huashang"]
        roi = (0, 700, 16000, 40800)
        split_target = (4, 8)
    elif guang_type in ['cb_tongzhou', 'cb_suidao']:
        # 长边切4块, 短边切3块
        split_targets = [(1, 4), (1, 3)]
        if guang_type == 'cb_tongzhou':
            rois = [(1200, 1000, 2100, 20900), (1200, 1000, 2100, 14900)]
        elif guang_type == 'cb_suidao':
            rois = [(1400, 700, 2500, 20800), (1480, 600, 2500, 14300)]
        defects = ["ignore", "dds-cb-pengshang", "dds-cb-huashang"]


    defcet_nums = [0]*len(defects)
    # data_list = [r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\黑线\银白色\0523']
    # save_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\cuted_dir'

    # data_list = [r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\【5】脏污异色\脏污异色分时拆分后']
    # save_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\分时明场\cuted_dir'

    data_list = [r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\腐蚀点\银白色\0524']
    save_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\cb_tz\cuted_dir'
    mkdir(save_dir)

    # path需要包含img和json
    for path in data_list:
        # 对物料在左在右进行划分
        left_dir = os.path.join(path, 'left')
        right_dir = os.path.join(path, 'right')
        long_dir = os.path.join(path, 'long')
        short_dir = os.path.join(path, 'short')
        if flag == 1:
            if guang_type in ["suidao", "tongzhou"]:
                # 隧道光的左右(1左2右)和同轴光的左右(2左1右)是相反的.
                split_left_right(path, left_dir, right_dir, guang=guang_type)
                names = path.split('\\')[-3:]
                cuted_dir = os.path.join(save_dir, names[0], names[1], names[2])
                mkdir(cuted_dir)
                help(left_dir, rois[1], split_target, cuted_dir)
                help(right_dir, rois[0], split_target, cuted_dir)
            elif guang_type in ['fsmc', 'fs']:
                names = path.split('\\')[-3:]
                cuted_dir = os.path.join(save_dir, names[0], names[1], names[2])
                mkdir(cuted_dir)
                help(path, roi, split_target, cuted_dir)
            elif guang_type in ['cb_suidao', 'cb_tongzhou']:
                split_AC_BD(path, long_dir, short_dir)
                names = path.split('\\')[-3:]
                cuted_dir = os.path.join(save_dir, names[0], names[1], names[2])
                mkdir(cuted_dir)
                help(long_dir, rois[0], split_targets[0], cuted_dir)
                help(short_dir, rois[1], split_targets[1], cuted_dir)

        elif flag == 2:
            tmp_dir = r'C:\Users\15974\Desktop\111'
            # merge(tmp_dir, rois[1])
            merge(tmp_dir, rois[0])

        elif flag == 3:
            cuted_dir = r'C:\Users\15974\Desktop\【4】DDS侧边划伤\【4】DDS侧边划伤\隧道光'
            js_paths = [os.path.join(cuted_dir, a) for a in os.listdir(cuted_dir) if '.json' in a]
            for js_path in js_paths:
                json_label_check(js_path, defects, defcet_nums)
            a = ''
            for ind, b in enumerate(defects):
                a += '{}: {}, '.format(b, defcet_nums[ind])
            print(a)







