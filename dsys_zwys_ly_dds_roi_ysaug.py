# coding=utf-8
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


roi_fs_mc = (1500, 300, 15000, 20600)
rois = [(1200, 600, 8192, 21000), (0, 500, 6800, 20800)] 
dir_ = r'C:\Users\15974\Desktop\roitest'
imgs = [a for a in os.listdir(dir_) if '.bmp' in a]
for img_name in imgs:
    if img_name.split('-')[1] == '2':
        # 同轴的物料在右边
        roi = rois[1]
        img = Image.open(os.path.join(dir_, img_name))
        img = np.asarray(img)
        img = img[roi[1]:roi[3], roi[0]:roi[2]]
        cv2.imwrite(os.path.join(dir_, 'slim_'+img_name), img)





# img_path = r'C:\Users\15974\Desktop\roitest\dds\2-3-9.bmp'
# roi_fs = (0, 700, 16000, 40800)
# img = Image.open(img_path)
# img = np.asarray(img)
# img = img[roi_fs[1]:roi_fs[3], roi_fs[0]:roi_fs[2]]
# cv2.imwrite(r'C:\Users\15974\Desktop\roitest\dds\slim_img.bmp', img)

