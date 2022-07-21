# coding=utf-8
# dst_Otsu2 可以理解为就是我们想要的apple-logo外扩一点点的,AA区域.
# 用 dst_Otsu2 or dst_Otsu4  都可.
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\apple\image.jpg'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
image = image[1185:20000, 2500:]
otsuThe, maxValue = 0, 255  # otsuThe=46
otsuThe, dst_Otsu = cv2.threshold(image, 1, maxValue, cv2.THRESH_OTSU)
dst_Otsu = cv2.bitwise_not(dst_Otsu)

# 检测出的apple-logo边上还是有点黑点, so先腐蚀(去除黑点)再膨胀(外扩白像素.)
kernel = np.ones((50, 50), dtype=np.uint8)
dst_Otsu1 = cv2.erode(dst_Otsu, kernel, iterations=1)
kernel = np.ones((30, 30), dtype=np.uint8)
dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 5)  
dst_Otsu3 = cv2.dilate(dst_Otsu1, kernel, 6) 
kernel_ = np.ones((40, 40), dtype=np.uint8)
dst_Otsu4 = cv2.dilate(dst_Otsu1, kernel_, 5)  

# plt.subplot(131) 
# plt.xlabel('org')
# plt.imshow(dst_Otsu)
# plt.subplot(132) 
# plt.xlabel('pz')
# plt.imshow(dst_Otsu1)
# plt.subplot(133) 
# plt.xlabel('pz+fs')
# plt.imshow(dst_Otsu2)
# plt.show()

# 存下applle-logo的mask图.
cv2.imwrite(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir\1.jpg', dst_Otsu2)
cv2.imwrite(r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir\2.jpg', dst_Otsu4)

# h, w = dst_Otsu.shape[:2]
# for i in range(h):
#     for j in range(w):

org = np.sum(dst_Otsu)
org_en = np.sum(dst_Otsu1)
org_en_dil = np.sum(dst_Otsu2)
print("org: {}, end: {}, end+dil: {}".format(org, org_en, org_en_dil))
print(np.sum(dst_Otsu3), np.sum(dst_Otsu4))

# 拟合外接圆
# contours, _ = cv2.findContours(dst_Otsu2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)):
#     (x, y), radius = cv2.minEnclosingCircle(contours[i])  
#     print((x, y), radius)
#     cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 20)
#     cv2.imwrite('./{}.jpg'.format(i), image)

contours, _ = cv2.findContours(dst_Otsu4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    points = cnt.reshape(cnt.shape[0], -1)
    rect = cv2.boundingRect(points)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[3]+rect[1]), (0, 255, 0), 10)
    cv2.imwrite('./{}.jpg'.format(i), image)