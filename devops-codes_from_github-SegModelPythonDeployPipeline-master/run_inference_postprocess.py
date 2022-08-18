# coding=utf-8
import os

import cv2
import numpy as np
from PIL import Image

from inference import OnnxInference
from painter import PainterBase
from settings import Kersen

Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import cv_imwrite


class CutSubimgsAndMergeSubimgs:
    def __init__(self, roi_param, split_strget, H_full, W_full):
        self.roi = roi_param
        self.split_target = split_strget
        self.H_full, self.W_full = H_full, W_full

    def roi_cut_imgtest(self, full_img_path, cuted_dir):
        if not os.path.exists(cuted_dir):
            os.makedirs(cuted_dir)
        basename = os.path.basename(full_img_path)
        name = basename.split('.')[0]
        img = Image.open(full_img_path)
        img = np.asarray(img)
        img_roied = img[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        h, w = img_roied.shape[:2]
        sub_h, sub_w = h // self.split_target[1], w // self.split_target[0]
        for i in range(self.split_target[0]):
            for j in range(self.split_target[1]):
                sub_img = img_roied[sub_h * j: sub_h * (j + 1), sub_w * i: sub_w * (i + 1)]
                sub_name = name.split('.')[0] + '_{}_{}.bmp'.format(j, i)
                sub_img_bgr = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(cuted_dir, sub_name), sub_img_bgr)

        return sub_h, sub_w

    def merge(self, subimg_ress, h_, w_):
        full_img = np.zeros((h_ * self.split_target[1], w_ * self.split_target[0], 3))
        full_ = np.zeros((self.W_full, self.H_full, 3))
        for i in range(self.split_target[0]):
            for j in range(self.split_target[1]):
                img = subimg_ress[j][i]
                full_img[h_ * j:h_ * (j + 1), w_ * i:w_ * (i + 1)] = img
        full_[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :] = full_img

        return full_


class InferencePostprocess:
    def __init__(self, inference_settings, postprocess_settings):
        self.inference_settings = inference_settings
        self.postprocess_settings = postprocess_settings
        self.painters = []

        self.roi = self.inference_settings["roi"]
        self.split_target = self.inference_settings["split_target"]
        self.scale_w, self.scale_h = self.inference_settings["scale_w_scale_h"][:2]
        self.H_full, self.W_full = self.inference_settings["H_full_W_full"][:2]
        self.input_size = self.inference_settings["input_size"]

    def for_each_image(self, root_dir):
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
                    yield os.path.join(root, file)

    def init_painters(self, paint_in_one_image=False):
        out_dir = self.postprocess_settings["dir_out"]
        label_map = self.postprocess_settings["label_map"]
        bg_index = self.postprocess_settings["bg_index"]
        ignore_index = self.postprocess_settings["ignore_index"]
        color_map = PainterBase.random_colors(label_map, bg_index, ignore_index)

        painters_config = self.postprocess_settings["painters"]
        out_size_type = self.postprocess_settings.get("out_size_type", "roi")

        for painter in painters_config:
            self.painters.append(painter(out_dir, label_map, bg_index, ignore_index, color_map, paint_in_one_image,
                                         out_size_type=out_size_type))

    def base_postprocess(self, map_, predict, filters):
        bboxs, confluences, areas = [], [], []
        num_class = predict.shape[1]
        mask_ = map_[0, :, :]
        temo_predict = np.zeros(mask_.shape)
        mask_map = np.max(predict[0, :, :, :], axis=0)

        for cls in range(1, num_class):
            mask = np.array(mask_ == cls, np.uint8)
            cc_output = cv2.connectedComponentsWithStats(mask, 8)
            num_contours, cc_labels, cc_stats = cc_output[:3]
            # for each contour
            for label in range(1, num_contours):
                x, y = cc_stats[label, cv2.CC_STAT_LEFT], cc_stats[label, cv2.CC_STAT_TOP]
                w, h = cc_stats[label, cv2.CC_STAT_WIDTH], cc_stats[label, cv2.CC_STAT_HEIGHT]
                area = cc_stats[label, cv2.CC_STAT_AREA]
                temp = np.array(cc_labels == label, np.uint8)
                score_temp = temp * mask_map
                mean_score = np.sum(score_temp) / area
                # confluence and area filter 
                if (area >= filters[0][cls]) and (mean_score >= filters[1][cls]):
                    temo_predict += temp * label
                    confluences.append(mean_score)
                    bboxs.append([x, y, x + w, y + h])
                    areas.append(area)

        return temo_predict, bboxs, confluences, areas

    def subimg_painter(self, subimg_reses, sub_imname, w_, h_, map_, scores, boxes, areas):
        img = cv2.imread(sub_imname)
        img = cv2.resize(img, (self.input_size[0], self.input_size[1]))
        j, i = int(os.path.basename(sub_imname).split('_')[1]), int(os.path.basename(sub_imname).split('_')[2][0])
        m = map_.astype(np.uint8)
        r, c = m.shape[:2]
        mask_vis = np.zeros((r, c, 3), dtype=np.uint8)
        mask_vis[:, :, 0] = (m & 1) << 7 | (m & 8) << 3
        mask_vis[:, :, 1] = (m & 2) << 6 | (m & 16) << 2
        mask_vis[:, :, 2] = (m & 4) << 5
        if len(scores):
            for ind, box in enumerate(boxes):
                cv2.rectangle(mask_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                box = [box[:2], box[2:]]
                box1 = [
                    [int(self.scale_w * a[0]) + w_ * i + self.roi[0], int(self.scale_h * a[1]) + h_ * j + self.roi[1]]
                    for a in box]
                text = '{}, '.format(np.round(scores[ind], 2))
                text += ''.join(str(a) + ',' for a in box1)
                text += '{}'.format(int(areas[ind] * self.scale_w * self.scale_h))
                cv2.putText(mask_vis, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
        sub_inference_img = cv2.resize(img_save, (w_, h_))
        print('save sub inference result ~.')
        subimg_reses[j][i] = sub_inference_img

    def run_inference_postprocess(self):
        model_onnx = OnnxInference(**self.inference_settings)
        image_dir = self.inference_settings["image_dir"]
        out_dir = self.postprocess_settings["dir_out"]
        post_pipeline = self.postprocess_settings["post_pipeline"]
        filters = [post_pipeline["PostFilterByPredictScore"], post_pipeline["PostFilterByArea"]]
        # self.init_painters()

        #  init CutSubimgsAndMergeSubimgs
        cub_and_merge = CutSubimgsAndMergeSubimgs(self.roi, self.split_target, self.H_full, self.W_full)

        for image_path in self.for_each_image(image_dir):
            prefix_dir = os.path.dirname(image_path)
            cuted_dir = os.path.join(prefix_dir, 'subimgs')
            # roi and cut subimgs
            h_, w_ = cub_and_merge.roi_cut_imgtest(image_path, cuted_dir)
            sub_img_paths = [os.path.join(cuted_dir, a) for a in os.listdir(cuted_dir)]

            # inference sub_img
            subimg_reses = [[0] * self.split_target[0] for j in range(self.split_target[1])]
            for sub_img_path in sub_img_paths:
                onnx_predict_index, onnx_predict_score = model_onnx.forward(sub_img_path)
                # subimg_base_postprocess
                map_, bboxs, confluences, areas = self.base_postprocess(onnx_predict_index, onnx_predict_score, filters)
                # subimg_painter
                self.subimg_painter(subimg_reses, sub_img_path, w_, h_, map_, confluences, bboxs, areas)
            full_img_result = cub_and_merge.merge(subimg_reses, h_, w_)

            # save res image
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            out_filename = "res_{}{}".format(name, ext)
            out_path = os.path.join(out_dir, out_filename)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            print(out_path)
            cv_imwrite(full_img_result, out_path)
            # cv2.imwrite(out_path, full_img_result)


if __name__ == "__main__":
    inference_settings, postprocess_settings = Kersen().get_inference_and_postprocess_settings()
    ip = InferencePostprocess(inference_settings, postprocess_settings)
    ip.run_inference_postprocess()
