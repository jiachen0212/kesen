import cv2
import numpy as np
import os


def cv_imread_by_np(filePath, clr_type=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), clr_type)

    return cv_img


def cv_imwrite(image, dst):
    name = os.path.basename(dst)
    cuted_dirfix, postfix = os.path.splitext(name)[:2]
    cv2.imencode(ext=postfix, img=image)[1].tofile(dst)


if __name__ == "__main__":

    img = cv_imread_by_np(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\漏杀\CCD3\24KSA0000000876225\漏杀.PNG')
    print(img.shape)

    cv_imwrite(img, r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\漏杀\CCD3\24KSA0000000876225\漏杀1.PNG')

