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
    

def cv_imread_by_np(filePath, clr_type=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), clr_type)

    return cv_img

def cv_imwrite(image, dst):
    name = os.path.basename(dst)
    cuted_dirfix, postfix = os.path.splitext(name)[:2]
    cv2.imencode(ext=postfix, img=image)[1].tofile(dst)

if __name__ == "__main__":

    # merge_imgs()
    # merge_jsons()
    # roi = (0, 2341, 7081, 21465)
    # roi = (0, 2341, 3909, 13323)
    # image = Image.open(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\train_dir\image.bmp')
    # image = np.asarray(image)
    # roi_img = image[roi[1]:roi[3], roi[0]:roi[2]]
    # roi_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\train_dir\template.bmp', roi_img)

    # roi = (1405, 2285, 8192, 21400)
    # roi = (1405, 2285, 6809, 21400)
    # image = Image.open(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\train_dir\image.bmp')
    # image = np.asarray(image)
    # roi_img = image[roi[1]:roi[3], roi[0]:roi[2]]
    # roi_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\train_dir\template.bmp', roi_img)


    # root = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\腐蚀点\银白色\0524'
    # ims = ['1-1-8.bmp', '1-2-8.bmp']
    # for im in ims:
    #     path = os.path.join(root, im)
    #     img = Image.open(path)
    #     img = np.asarray(img)
    #     temp = [a for a in img[int(img.shape[0]*2.2//3)][0]]
    #     if np.sum(temp) > 0:
    #         # 'left'
    #        roi = (0, 2100, 7256, 21640)
    #        roi_img = img[roi[1]: roi[3], roi[0]:roi[2]]
    #        cv_imwrite(roi_img, r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\1.jpg')

    path_ = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\infrence_dir'
    im = r'25-1-3.bmp'
    js = r'25-1-3.json'
    img = np.asarray(Image.open(os.path.join(path_, im)))
    jss = json.load(open(os.path.join(path_, js), 'r'))
    p1, p2 = jss['area_points'][0], jss['area_points'][2]
    roi = p1 + p2
    # point_color = (255, 255, 255) # BGR
    # thickness = 10
    # lineType = 8
    # cv2.rectangle(img, p1, p2, point_color, thickness, lineType)
    # cv_imwrite(img, r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\infrence_dir\1.jpg')
    print(roi)
