import copy
import json
import os
import numpy as np 
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None  # 解除最大图像尺寸限制


def crop_left_edge(points, x):
    # 处理左边切割
    n = len(points)
    feedback = []
    for i in range(n):
        s_point = points[i % n]
        e_point = points[(i + 1) % n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[0] >= x and e_point[0] >= x:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[0] >= x and e_point[0] < x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x - e_point[0]) / float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x, y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[0] < x and e_point[0] >= x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x - e_point[0]) / float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x, y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback


def crop_right_edge(points, x):
    # 处理右边切割
    n = len(points)
    feedback = []
    for i in range(n):
        s_point = points[i % n]
        e_point = points[(i + 1) % n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[0] <= x and e_point[0] <= x:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[0] <= x and e_point[0] > x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x - e_point[0]) / float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x, y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[0] > x and e_point[0] <= x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x - e_point[0]) / float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x, y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback


def crop_upper_edge(points, y):
    # 处理上边切割
    n = len(points)
    feedback = []

    for i in range(n):
        s_point = points[i % n]
        e_point = points[(i + 1) % n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[1] >= y and e_point[1] >= y:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[1] >= y and e_point[1] < y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[1] < y and e_point[1] >= y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback


def crop_lower_edge(points, y):
    # 处理下边切割
    n = len(points)
    feedback = []

    for i in range(n):
        s_point = points[i % n]
        e_point = points[(i + 1) % n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[1] <= y and e_point[1] <= y:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[1] <= y and e_point[1] > y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[1] > y and e_point[1] <= y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback


def transfer_one_polygon(polygon, crop_area):
    points = copy.deepcopy(polygon)
    left, upper, right, lower = crop_area

    # 通过Sutherland-Hodgman算法 实现四边形裁剪任意多边形
    points = crop_left_edge(points, left)
    points = crop_right_edge(points, right)
    points = crop_upper_edge(points, upper)
    points = crop_lower_edge(points, lower)

    # 坐标偏移, 将基坐标转移到[0,0]点
    height_limit = lower - upper
    width_limit = right - left
    for index, point in enumerate(points):
        x = point[0] - left
        x = x if x >= 0 else 0
        x = x if x < width_limit else width_limit - 1
        y = point[1] - upper
        y = y if y >= 0 else 0
        y = y if y < height_limit else height_limit - 1
        points[index] = [x, y]
    return points


def crop_area_with_seg_annotation(img, crop_area, cropped_img_name, anno_template):
    left, upper, right, lower = crop_area

    cropped = img.crop(crop_area)

    # 处理多边形标注
    if anno_template:
        out_path_prefix, image_name = os.path.split(cropped_img_name)
        cropped_annotation = copy.deepcopy(anno_template)
        shapes = cropped_annotation.get("shapes", [])
        new_shapes = []
        for shape in shapes:
            if shape.get("shape_type", None) == "polygon":
                new_points = transfer_one_polygon(shape.get("points", []), crop_area)
                if new_points:
                    for index, point in enumerate(new_points):
                        new_points[index] = [point[0], point[1]]
                    shape["points"] = new_points
                    new_shapes.append(shape)

        cropped_annotation["shapes"] = new_shapes
        cropped_annotation["imagePath"] = os.path.split(cropped_img_name)[1]
        cropped_annotation["imageData"] = None
        cropped_annotation["imageHeight"] = int(lower - upper)
        cropped_annotation["imageWidth"] = int(right - left)
        if new_shapes:  # 子图中包含缺陷
            out_path_prefix = os.path.join(out_path_prefix, "ng")
        else:  # 子图中不包含缺陷
            out_path_prefix = os.path.join(out_path_prefix, "ok")

        if not os.path.isdir(out_path_prefix):
            os.makedirs(out_path_prefix)
        out_image_path = os.path.join(out_path_prefix, image_name)
        cropped.save(out_image_path)
        anno_filename = os.path.splitext(out_image_path)[0] + ".json"
        with open(os.path.join(anno_filename), "w", encoding="utf-8") as writer:
            json.dump(cropped_annotation, writer)
    else:
        cropped.save(cropped_img_name)


def split_image(path_image, split_target, out_dir=None, roi=None):
    img = Image.open(path_image)
    w, h = img.size
    if roi:
        w, h = min(roi[2] - roi[0], w), min(roi[3] - roi[1], h)

    target_w, target_h = split_target
    step_w = w / target_w
    step_h = h / target_h

    pre_name, ext = os.path.splitext(path_image)
    anno_template = None
    if os.path.isfile(pre_name + ".json"):
        with open(pre_name + ".json", "r", encoding="utf-8") as reader:
            anno_template = json.load(reader)

    for i in range(target_w):
        for j in range(target_h):
            # 处理输出图像路径&名称
            if not out_dir:
                pre, ext = os.path.splitext(path_image)
            else:
                image_name_ext = os.path.split(path_image)[1]
                image_name = os.path.splitext(image_name_ext)[0]
                pre = os.path.join(out_dir, image_name)
            cropped_img_name = "{}_{}_{}_{}".format(pre, i, j, ext)

            # 处理子图裁剪
            left = step_w * i + roi[0] if roi else step_w * i
            upper = step_h * j + roi[1] if roi else step_h * j
            right = step_w * (i + 1) + roi[0] if roi else step_w * (i + 1)
            lower = step_h * (j + 1) + roi[1] if roi else step_h * (j + 1)
            if roi:
                if right > roi[2]:
                    right = roi[2]
                if lower > roi[3]:
                    lower = roi[3]
            else:
                if right > w:
                    right = w
                if lower > h:
                    lower = h

            crop_area = (left, upper, right, lower)  # (left, upper, right, lower)
            # print(i,j,crop_area)
            crop_area_with_seg_annotation(img, crop_area, cropped_img_name, anno_template)


def split_img_dir(source_path, split_target, out_dir=None, roi=None):
    for root, dirs, files in os.walk(source_path):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
                tmp_dir_out = root.replace(source_path, out_dir)
                os.makedirs(tmp_dir_out, exist_ok=True)
                print(tmp_dir_out)
                split_image(os.path.join(root, filename), split_target, tmp_dir_out, roi)


def help(data, defects, js, cuts_dir):
    for ind, defect in enumerate(defects):
        if defect in js:
            line = '{},{}\n'.format(os.path.join(cuts_dir, js.split('.')[0]+'.jpg'), os.path.join(cuts_dir, js))
            print(line)
            data[ind].append(line)




def generate_train_test_txt(defects, out_dir, train_txt, test_txt):

    defect_lens = len(defects)
    # 按照各个缺陷类别划分数据集
    datas = [[] for a in range(defect_lens)]
    jss = os.listdir(out_dir)
    jss = [a for a in jss if '.json' in a]
    for js in jss:
        help(datas, defects, js, out_dir)

    trains, tests = [], []
    for data in datas:
        len_ = int(len(data)*0.8)
        trains.extend(data[:len_])
        tests.extend(data[len_:])
    for tr in trains:
        train_txt.write(tr)
    for te in tests:
        test_txt.write(te)



# if __name__ == "__main__":
def roi_cut_bins(roi, source_path, split_target):

    out_dir = os.path.join(source_path, 'cuts')
    split_img_dir(source_path, split_target, out_dir, roi)


    # 数据写入train, test .txt
    # train_txt = open('./train.txt', 'w')
    # test_txt = open('./test.txt', 'w')
    # defects = ['zangwu', 'huaheng', 'aotu', 'baidian']
    # out_dir1 = os.path.join(out_dir, 'ng')
    # generate_train_test_txt(defects, out_dir1, train_txt, test_txt)



def cv_imwrite(image, dst):
    name = os.path.basename(dst)
    _, postfix = os.path.splitext(name)[:2]
    cv2.imencode(ext=postfix, img=image)[1].tofile(dst)


def roi_cut_imgtest(img_path, roi, split_target, cuted_dir):
    basename = os.path.basename(img_path)
    name = basename.split('.')[0]
    img = Image.open(img_path)
    img = np.asarray(img)
    img_roied = img[roi[1]:roi[3], roi[0]:roi[2]]
    # cv_imwrite(img_roied, os.path.join(r'C:\Users\15974\Desktop\1', basename))
    h, w = img_roied.shape[:2]
    sub_h, sub_w = h//split_target[1], w//split_target[0]
    for i in range(split_target[0]):
        for j in range(split_target[1]):
            sub_img = img_roied[sub_h*j: sub_h*(j+1), sub_w*i: sub_w*(i+1)]
            sub_name = name.split('.')[0]+'_{}_{}.jpg'.format(j, i)
            sub_img_bgr = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img_bgr)
            cv_imwrite(sub_img_bgr, os.path.join(cuted_dir, sub_name))
            

if __name__ == "__main__":

    roi = (500, 1200, 7600, 15400)
    source_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\vimo算法联动\新博成氢电池\0706data\黑点1'
    split_target = (6, 14)
    img_paths = [os.path.join(source_path, a) for a in os.listdir(source_path)]
    cuted_dir = r'C:\Users\15974\Desktop\xbc_0706\1'
    for img_path in img_paths:
        roi_cut_imgtest(img_path, roi, split_target, cuted_dir)
