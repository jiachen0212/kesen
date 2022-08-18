import cv2
import numpy as np


## 读取图像，解决imread不能读取中文路径的问题
def cv_imread_by_np(filePath, clr_type=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), clr_type)
    return cv_img


def cv_imwrite(image, dst):
    cv2.imencode(ext='.png', img=image)[1].tofile(dst)
