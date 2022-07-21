# codoing=utf-8
import os
import cv2
import numpy as np

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

data_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\0523\黑线\银白色\条光3-4'
# res_dir = os.path.join(data_path, 'split')
res_dir = r'C:\Users\15974\Desktop\2'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

img_paths = [os.path.join(data_path, a) for a in os.listdir(data_path)]
for path_ in img_paths:
    basename = os.path.basename(path_)
    name = basename.split('.')[0]
    img = Image.open(path_)
    img_ = np.asarray(img)
    h, w = img_.shape[:2]
    h_ = h//2
    img1, img2 = img_[:h_, :], img_[h_:, :]
    save_path1 = os.path.join(res_dir, name+'_0.bmp')
    print(save_path1)
    save_path2 = os.path.join(res_dir, name+'_1.bmp')
    cv2.imwrite(save_path1, img1)
    cv2.imwrite(save_path2, img2)
