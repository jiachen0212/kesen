# coding=utf-8
from datetime import date
import os
from tkinter import image_names

import scipy as sp
from torch import full_like
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from debug import help
import shutil 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from locate.locate import main_fun
from speedup_sdk import sdk_fun 

def clean_dirs_test_tune_data(dir_):
    defect_dirs = [os.path.join(dir_, a) for a in os.listdir(dir_) if '.bmp' not in a]
    for base_dir in defect_dirs:
        im_names = [a for a in os.listdir(base_dir) if '.bmp' in a]
        for name in im_names:
            temp_path = os.path.join(base_dir, name)
            shutil.copy(temp_path, os.path.join(dir_, name))
        shutil.rmtree(base_dir)

def roicut_test_tune_data(dir_, left_dir, right_dir, split_target, js_dir=None, train_dir=None, label_json=None, left_roi_js=None, right_roi_js=None):
    # im_name: A180_KSA0000000875335_Snow_Station4_Linear_Tunnel_1_2022_07_09_21_07_50_638_RC_N_Ori.bmp
    im_names = [a for a in os.listdir(dir_) if '.bmp' in a]
    mkdir(left_dir)
    mkdir(right_dir)
    for im_name in im_names:
        sps = im_name.split('_')
        assert len(sps) == 17
        assert sps[6] in ['1', '2']
        left_of_right = sps[6]
        org = os.path.join(dir_, im_name)
        org_js = os.path.join(dir_, im_name.split('.')[0]+'.json')
        if left_of_right == '1':  # 物料在右
            shutil.copy(org, os.path.join(right_dir, im_name))  
            if os.path.exists(org_js):
                shutil.copy(org_js, os.path.join(right_dir, im_name.split('.')[0]+'.json'))  
                os.remove(org_js)
        else:
            shutil.copy(org, os.path.join(left_dir, im_name))
            if os.path.exists(org_js):
                shutil.copy(org_js, os.path.join(left_dir, im_name.split('.')[0]+'.json'))  
                os.remove(org_js)
        os.remove(org)
        roi_left = main_fun(js_dir, left_dir, flag='left', roi_vis_path=False, train_dir=train_dir)
        roi_right = main_fun(js_dir, right_dir, flag='right', roi_vis_path=False, train_dir=train_dir)
        # 如果有label-json, 则做img+json的切割
    if label_json:
        save_dir = os.path.join(dir_, 'cuted')
        help(left_dir, roi_left, split_target, save_dir)
        help(right_dir, roi_right, split_target, save_dir)

    # roi_left, roi_right json 落盘
    with open(left_roi_js, "w", encoding='utf-8') as fp:
        json.dump(roi_left, fp,ensure_ascii=False,indent = 4)
    with open(right_roi_js, "w", encoding='utf-8') as fp:
        json.dump(roi_right, fp,ensure_ascii=False,indent = 4)
    


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


def defect_number(left_dir):
    js_paths = [os.path.join(left_dir, a) for a in os.listdir(left_dir) if '.json' in a]
    for js_path in js_paths:
        json_label_check(js_path, defects, defcet_nums)
    a = ''
    for ind, b in enumerate(defects):
        a += '{}: {}, '.format(b, defcet_nums[ind])

    print(a)


if __name__ == '__main__':

    # flag1 处理前线传回的数据, 拆分出左右, 模板匹配定位得到imgs_rois
        # 如果有标注的jsons, label_json=True, 会对img+json做roi和cutbins切割.

    # flag2 calculate defect_nums

    # flag3 本地导入onnx, 并做批量数据test.
        # 测试结果存为.jpg落盘

    flag = 1

    guang_type = 'suidao'  
    if guang_type in ['suidao']:
        defects = ["heixian", "fushidian", "huichen", "zangwu", "ignore", "znagwu"]
        split_target = (2, 4) 
    defcet_nums = [0]*len(defects)
    
    # data path 
    test_tun_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\98pcs 黑线原图'
    left_dir, right_dir = os.path.join(test_tun_path, 'left'), os.path.join(test_tun_path, 'right')
    left_roi_js, right_roi_js = './left_roi.json', './right_roi.json'
    # infernece_res_save_dir
    res_dir = os.path.join(test_tun_path, 'inference_res')
    
    if flag == 1:
        label_json = False
        clean_dirs_test_tune_data(test_tun_path)
        js_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate'
        train_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\train_dir'
        roicut_test_tune_data(test_tun_path, left_dir, right_dir, split_target, js_dir=js_dir, train_dir=train_dir, label_json=label_json, left_roi_js=left_roi_js, right_roi_js=right_roi_js)
    
    elif flag == 2:
        defect_number(left_dir)
        defect_number(right_dir)
    
    elif flag == 3:
        onnx_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\station3_20220626_suidao_2000iter.onnx'
        mkdir(res_dir)
        left_roi_jsdata = json.load(open(left_roi_js, 'r'))
        right_roi_jsdata = json.load(open(right_roi_js, 'r'))
        # 分别inference left_dir, right_dir 
        for im_name, l_roi in left_roi_jsdata.items():
            img_path = os.path.join(left_dir, im_name)
            sdk_fun(onnx_path, img_path, l_roi)
        for im_name, r_roi in right_roi_jsdata.items():
            img_path = os.path.join(right_dir, im_name)
            sdk_fun(onnx_path, img_path, r_roi, res_dir)
