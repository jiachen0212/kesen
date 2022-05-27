# coding=utf-8
import os

from torch import le
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from debug import help
import shutil 
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# def get_dir(dir1):
#     if os.path.exists(dir1):
#         if not os.path.isfile(dir1):
#             dir1_s = [os.path.join(dir1, a) for a in os.listdir(dir1)]
#             for dir_ in dir1_s:
#                 if os.path.isfile(dir_):
#                     return dir1
#                 else:
#                     return get_dir(dir_)


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
                print(os.path.join(cuted_dir, sub_name))
                cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img)


def merge(sub_imgs_dir, roi):
    full_img = np.zeros((4885*4, 3628*2, 3))
    full_ = np.zeros((22000, 8192, 3))
    for i in range(2):
        for j in range(4):
            path = os.path.join(sub_imgs_dir, '1-1-5_{}_{}.jpg'.format(j, i))
            img = cv2.imread(path)  # 竖直_水平
            h,w = img.shape[:2]
            full_img[h*j:h*(j+1), w*i:w*(i+1)] = img
    
    full_[roi[1]:roi[3], roi[0]:roi[2],:] = full_img
    cv2.imwrite(os.path.join(sub_imgs_dir, '2.jpg'), full_)
    return full_


if __name__ == '__main__':
    base_dirs = [r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\黑线\银白色', r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\腐蚀点\银白色']  
    save_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\cuted_dir'
    mkdir(save_dir)

    # flag0 img roi_cut_test, flag1 切割img和json, flag2 merge test
    flag = 2

    split_target = (2, 4)
    rois = [(1200, 2026, 8192, 21650), (0, 2100, 7256, 21640)] 

    for base_dir in base_dirs:
        data_list = [os.path.join(base_dir, a) for a in os.listdir(base_dir)]
        # path需要包含img和json
        for path in data_list:
            # 对物料在左在右进行划分
            left_dir = os.path.join(path, 'left')
            right_dir = os.path.join(path, 'right')
            split_left_right(path, left_dir, right_dir)
            if flag == 1:
                names = path.split('\\')[-3:]
                cuted_dir = os.path.join(save_dir, names[0], names[1], names[2])
                mkdir(cuted_dir)
                help(left_dir, rois[1], split_target, cuted_dir)
                help(right_dir, rois[0], split_target, cuted_dir)
            elif flag == 0:
                cuted_dir = os.path.join(r'C:\Users\15974\Desktop\ks', 'cuted')
                mkdir(cuted_dir)
                roi_cut_imgtest(left_dir, rois[1], split_target, cuted_dir)
            elif flag ==2:
                sub_imgs_dir = r'C:\Users\15974\Desktop\ks\cuted'
                merge(sub_imgs_dir, rois[1])
