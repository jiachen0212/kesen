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


def clean_dirs_test_tune_data(dir_):
    ccds = ['CCD3', 'CCD4']
    defect_dirs = [os.path.join(dir_, a) for a in os.listdir(dir_)]
    for defect_dir in defect_dirs:
        for ccd in ccds:
            base_dir = os.path.join(defect_dir, ccd)
            im_dirs = os.listdir(base_dir)
            for im_dir in im_dirs:
                temp_path = os.path.join(base_dir, im_dir)
                ims = [a for a in os.listdir(temp_path) if '.bmp' in a]
                for a in ims:
                    shutil.copy(os.path.join(temp_path, a), os.path.join(defect_dir, a))
            shutil.rmtree(base_dir)


def roicut_test_tune_data(dir_, rois, split_target):
    defect_dirs = [os.path.join(dir_, a) for a in os.listdir(dir_)]
    for defect_dir in defect_dirs:
        save_dir = os.path.join(defect_dir, 'roicut')
        left_dir, right_dir = os.path.join(defect_dir, 'left'), os.path.join(defect_dir, 'right')
        split_left_right(defect_dir, left_dir, right_dir, guang=guang_type)
        help(left_dir, rois[1], split_target, save_dir)
        help(right_dir, rois[0], split_target, save_dir)
        shutil.rmtree(left_dir)
        shutil.rmtree(right_dir)


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
        cuted_dirfix, postfix = os.path.splitext(file)[:2]
        js_file = cuted_dirfix +'.json'
        
        org = os.path.join(path, file) 
        img_pil = Image.open(org)
        img = np.asarray(img_pil)
        temp = [a for a in img[int(img.shape[0]*2.2//3)][0]]
        if np.sum(temp) > 0:
            # 'left'
            org1, std1 = os.path.join(path, file), os.path.join(left_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(left_dir, js_file)
        elif np.sum(temp) == 0:
            # 'right'
            org1, std1 = os.path.join(path, file), os.path.join(right_dir, file)
            org2, std2 = os.path.join(path, js_file), os.path.join(right_dir, js_file)

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




if __name__ == '__main__':

    # flag1 split left right, then cut sub_bins.   
    # flag2 calculate defect nums

    guang_type = 'suidao'  
    flag = 3

    if guang_type in ['suidao']:
        rois = [(1200, 2026, 8192, 21650), (0, 2100, 7256, 21640)]
        defects = ["heixian", "fushidian", "huichen", "zangwu", "ignore", "znagwu"]
        split_target = (2, 4) 

    defcet_nums = [0]*len(defects)

    path = '/newdata/jiachen/data/kesen/base_suidao'
    save_dir = os.path.join(path, 'roi_cuted')
    mkdir(save_dir)

    # path需要包含img和json
    # 对物料在左在右进行划分
    left_dir = os.path.join(path, 'left')
    right_dir = os.path.join(path, 'right')
    long_dir = os.path.join(path, 'long')
    short_dir = os.path.join(path, 'short')
    if flag == 1:
        if guang_type in ["suidao", "tongzhou"]:
            # 隧道光的左右(1左2右)和同轴光的左右(2左1右)是相反的.
            split_left_right(path, left_dir, right_dir, guang=guang_type)
            help(left_dir, rois[1], split_target, save_dir)
            help(right_dir, rois[0], split_target, save_dir)
            shutil.rmtree(left_dir)
            shutil.rmtree(right_dir)

    elif flag == 2:
        js_paths = [os.path.join(save_dir, a) for a in os.listdir(save_dir) if '.json' in a]
        for js_path in js_paths:
            json_label_check(js_path, defects, defcet_nums)
    
    elif flag == 3:
        test_tun_path = '/newdata/jiachen/data/kesen/suidao_test_tune'
        # clean_dirs_test_tune_data(test_tun_path)
        roicut_test_tune_data(test_tun_path, rois, split_target)



