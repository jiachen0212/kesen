import os

# dir_ = r'C:\Users\15974\Desktop\111'
# names = os.listdir(dir_)
# for name in names:
#     new_name = name[:5]+name[7:] 
#     os.rename(os.path.join(dir_, name), os.path.join(dir_, new_name))


import cv2
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# path_ = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\黑线\银白色\0523\left\2-1-9.bmp'
# img = Image.open(path_)
# img = np.asarray(img)
# indexs = (4694, 7080, 4930, 8520)
# heixian_roi = img[indexs[1]:indexs[3], indexs[0]:indexs[2]]
# cv2.imwrite('./heixian_roi.bmp', heixian_roi)

def bmp2jpg():
    path = r'C:\Users\15974\Desktop\kesen\1.bmp'
    img = Image.open(path)
    img = np.asarray(img)
    cv2.imwrite(r'C:\Users\15974\Desktop\kesen\1.jpg', img)



# path = r'C:\Users\15974\Desktop\1.jpg'
# img = Image.open(path)
# img = np.asarray(img)
# roi = (1500, 300, 15000, 20600)
# img_roied = img[roi[1]:roi[3], roi[0]:roi[2]]
# cv2.imwrite(r'C:\Users\15974\Desktop\kesen\1_roi.jpg', img_roied)




def roi_cut_imgtest(dir_, name, roi, split_target, cuted_dir):

    img_path = os.path.join(dir_, name)
    img = Image.open(img_path)
    img = np.asarray(img)
    img_roied = img[roi[1]:roi[3], roi[0]:roi[2]]
    cv2.imwrite(os.path.join(cuted_dir, name), img_roied)
    # h, w = img_roied.shape[:2]
    # sub_h, sub_w = h//split_target[1], w//split_target[0]
    # for i in range(split_target[0]):
    #     for j in range(split_target[1]):
    #         sub_img = img_roied[sub_h*j: sub_h*(j+1), sub_w*i: sub_w*(i+1)]
    #         sub_name = name.split('.')[0]+'_{}_{}.jpg'.format(j,i)
    #         cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img)

guang_type = 'cb_suidao'   # cb_tongzhou

cuted_dir = r'C:\Users\15974\Desktop\1'
split_target = (1, 4) 

if guang_type == 'cb_tongzhou':
    rois = [(1200, 1000, 2100, 20900), (1200, 1000, 2100, 14900)]
elif guang_type == 'cb_suidao':
    rois = [(1400, 700, 2500, 20800), (1480, 600, 2500, 14300)]

dir_ = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\dds侧边\DDS侧边碰伤（3）\同轴光'
imgs = [a for a in os.listdir(dir_) if '.bmp' in a]
for im in imgs:
    index_b = im.split('-')[-1][0]
    assert index_b in 'ABCD'
    if index_b in 'AC':  # 长边
        roi = rois[0]
    elif index_b in 'BD': # 短边
        roi = rois[1]
    roi_cut_imgtest(dir_, im, roi, split_target, cuted_dir)