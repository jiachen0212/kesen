from PIL import Image
import os
import json
import copy
import shutil
import cv2

Image.MAX_IMAGE_PIXELS = None # 解除最大图像尺寸限制

def crop_left_edge(points,x):
    # 处理左边切割
    n = len(points)
    feedback = []
    for i in range(n):
        s_point = points[i%n]
        e_point = points[(i+1)%n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[0]>=x and e_point[0]>=x:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[0]>=x and e_point[0]<x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x-e_point[0])/float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x,y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[0]<x and e_point[0]>=x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x-e_point[0])/float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x,y]
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
    points = crop_left_edge(points,left)
    points = crop_right_edge(points,right)
    points = crop_upper_edge(points,upper)
    points = crop_lower_edge(points,lower)

    # 坐标偏移, 将基坐标转移到[0,0]点
    height_limit = lower - upper
    width_limit = right - left
    for index,point in enumerate(points):
        x = point[0]-left
        x = x if x>=0 else 0
        x = x if x<width_limit else width_limit -1
        y = point[1]-upper
        y = y if y>=0 else 0
        y = y if y<height_limit else height_limit - 1
        points[index] = [x, y]
    return points

def crop_area_with_seg_annotation(img,crop_area,cropped_img_name,anno_template):
    left, upper, right, lower = crop_area

    cropped = img.crop(crop_area)
    cropped.save(cropped_img_name)

    # 处理多边形标注
    if anno_template:
        anno_filename = os.path.splitext(cropped_img_name)[0]+".json"
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
        if new_shapes:
            cropped_annotation["shapes"] = new_shapes
            cropped_annotation["imagePath"] = os.path.split(cropped_img_name)[1]
            cropped_annotation["imageData"] = None
            cropped_annotation["imageHeight"] = int(lower - upper)
            cropped_annotation["imageWidth"] = int(right - left)
            with open(anno_filename, "w", encoding="utf-8") as writer:
                json.dump(cropped_annotation, writer)


def split_image(path_image, split_target,out_dir=None, roi=None):
    img = Image.open(path_image)
    w, h = img.size
    if roi:
        w, h = min(roi[2]-roi[0], w), min(roi[3]-roi[1],h)

    target_w, target_h = split_target
    step_w = w/target_w
    step_h = h/target_h

    pre_name, ext = os.path.splitext(path_image)
    anno_template = None
    if os.path.isfile(pre_name+".json"):
        with open(pre_name+".json", "r", encoding="utf-8") as reader:
            anno_template = json.load(reader)

    for i in range(target_w):
        for j in range(target_h):
            # 处理输出图像路径&名称
            if not out_dir:
                pre, ext = os.path.splitext(path_image)
            else:
                image_name_ext = os.path.split(path_image)[1]
                image_name = os.path.splitext(image_name_ext)[0]
                pre = os.path.join(out_dir,image_name)
            cropped_img_name = "{}_{}_{}_{}".format(pre,i,j,ext)

            # 处理子图裁剪
            left = step_w * i + roi[0] if roi else step_w * i
            upper = step_h * j + roi[1] if roi else step_h * j
            right = step_w * (i+1) + roi[0] if roi else step_w * (i+1)
            lower = step_h * (j+1) + roi[1] if roi else step_h * (j+1)
            if roi:
                if right>roi[2]:
                    right = roi[2]
                if lower>roi[3]:
                    lower = roi[3]
            else:
                if right > w:
                    right = w
                if lower > h:
                    lower = h

            crop_area = (left, upper, right, lower) # (left, upper, right, lower)
            # print(i,j,crop_area)
            crop_area_with_seg_annotation(img,crop_area,cropped_img_name,anno_template)



def split_image1(path_image, split_target,img_roi, out_dir=None):
    img = Image.open(path_image)
    w, h = img.size
    roi = img_roi[os.path.basename(path_image)]
    print(roi, '1')
    w, h = min(roi[2]-roi[0], w), min(roi[3]-roi[1],h)

    target_w, target_h = split_target
    step_w = w/target_w
    step_h = h/target_h

    pre_name, ext = os.path.splitext(path_image)
    anno_template = None
    if os.path.isfile(pre_name+".json"):
        with open(pre_name+".json", "r", encoding="utf-8") as reader:
            anno_template = json.load(reader)

    for i in range(target_w):
        for j in range(target_h):
            # 处理输出图像路径&名称
            if not out_dir:
                pre, ext = os.path.splitext(path_image)
            else:
                image_name_ext = os.path.split(path_image)[1]
                image_name = os.path.splitext(image_name_ext)[0]
                pre = os.path.join(out_dir,image_name)
            cropped_img_name = "{}_{}_{}_{}".format(pre,i,j,ext)

            # 处理子图裁剪
            left = step_w * i + roi[0] if roi else step_w * i
            upper = step_h * j + roi[1] if roi else step_h * j
            right = step_w * (i+1) + roi[0] if roi else step_w * (i+1)
            lower = step_h * (j+1) + roi[1] if roi else step_h * (j+1)
            if roi:
                if right>roi[2]:
                    right = roi[2]
                if lower>roi[3]:
                    lower = roi[3]
            else:
                if right > w:
                    right = w
                if lower > h:
                    lower = h

            crop_area = (left, upper, right, lower) # (left, upper, right, lower)
            # print(i,j,crop_area)
            crop_area_with_seg_annotation(img,crop_area,cropped_img_name,anno_template)


def split_img_dir(source_path,split_target,out_dir=None,roi=None):
    for root,dirs,files in os.walk(source_path):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
                tmp_dir_out = root.replace(source_path,out_dir)
                os.makedirs(tmp_dir_out, exist_ok=True)
                split_image(os.path.join(root,filename),split_target,tmp_dir_out,roi)


def split_img_dir1(source_path,split_target,out_dir=None,roi=None):
    for root,dirs,files in os.walk(source_path):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
                tmp_dir_out = root.replace(source_path,out_dir)
                os.makedirs(tmp_dir_out, exist_ok=True)
                split_image1(os.path.join(root,filename),split_target,roi,tmp_dir_out)

def check_first_sub_img(out_dir):
    ims = os.listdir(out_dir)
    for im in ims:
        x, y = im.split('_')[-3], im.split('_')[-2]
        if x == y and x == '0':
            cv2.imshow('first_sub_im', cv2.imread(os.path.join(out_dir, im)))
            cv2.waitKey(100)


# if __name__ == "__main__":

def help(source_path, roi, split_target, out_dir):

    # roi = (918,0,7594, 7594)  # 横,纵 起终点.
    # split_target = (1, 2)
    if len(roi) == 4:
        split_img_dir(source_path,split_target,out_dir,roi)
    else:
        split_img_dir1(source_path,split_target,out_dir,roi)

    # check_first_sub_img(out_dir)
