# coding=utf-8
import os
from datetime import date
import os
from torch import full_like
from PIL import Image
from winds_roi_cut_imgs import merge
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import shutil 
import cv2
cv2.CV_IO_MAX_IMAGE_PIXELS = 200224000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json


def merge_imgs():
    dir_ = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\merge_3pics_multi_jsons'
    names = [a for a in os.listdir(dir_) if '.bmp' in a]
    gray_imgs = []
    for im in names:

        # cv2读图, 得到BGR格式的
        img = cv2.imread(os.path.join(dir_, im))
        print(img.shape)

        # Image 读图, 得到RGB格式的
        # img = Image.open(os.path.join(dir_, im))
        # img = np.asarray(img)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print(img.size)

        # img = img[:,:,0]

        # print(im, img[3333][3333])  # img3个通道的取值并不一样
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img_gray.shape)
        cv2.imwrite(os.path.join(dir_, 'gray_'+im), img_gray)  
        gray_imgs.append(img_gray)
    # cv2.imwrite(os.path.join(dir_, 'merged.bmp'), cv2.merge(gray_imgs))  

def merge_jsons():
    dir_ = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\merge_3pics_multi_jsons'
    jsons = [a for a in os.listdir(dir_) if '.json' in a]
    js_datas = []
    for js in jsons:
        js_path = os.path.join(dir_, js)
        js_data = json.load(open(js_path, 'r'))
        js_datas.append(js_data)
    merged_json = js_datas[0].copy()
    # 只需要修改'imagePath'和'shapes'
    merged_json['imagePath'] = 'merged.bmp'
    merged_shapes = []
    for js_data in js_datas:
        dicts = js_data['shapes']
        for dict_ in dicts:
            merged_shapes.append(dict_)
    merged_json['shapes'] = merged_shapes
    with open(os.path.join(dir_, 'merged.json'), "w", encoding="utf-8") as writer:
        json.dump(merged_json, writer, indent=4)
    

if __name__ == "__main__":

    merge_imgs()
    # merge_jsons()

