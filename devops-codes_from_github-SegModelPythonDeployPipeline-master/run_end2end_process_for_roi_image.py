import os

import numpy as np
from PIL import Image

from inference import OnnxInference
from painter import PainterBase
from preprocess import split_image
from settings import Kersen, Settings
from utils import for_each_image, affine_labelme_annotation

Image.MAX_IMAGE_PIXELS = None  # 解除最大图像尺寸限制


class End2End:
    def __init__(self, preprocess_settings, inference_settings, postprocess_settings):
        self.preprocess_settings = preprocess_settings
        self.original_image_name_path_dir = dict()
        self.inference_settings = inference_settings
        self.postprocess_settings = postprocess_settings
        self.painters = []
        self.image_name_path_dict = None
        self.split = None

        self.split = self.preprocess_settings["split"]
        self.pre_out_dir = self.preprocess_settings["dir_out"]

        self.init_painters(paint_in_one_image=self.postprocess_settings.get("post_process_settings", False))

    def run_end2end(self):
        print("Preprocessing {}".format(self.preprocess_settings["dir_in"]))
        model_onnx = OnnxInference(**self.inference_settings)
        post_pipeline = self.postprocess_settings["post_pipeline"]

        for real_path in Settings.filter_each_image(self.preprocess_settings):
            # do preprocess, roi & split
            print("\tProcessing {}".format(real_path))
            if not os.path.isdir(self.preprocess_settings["dir_out"]):
                os.makedirs(self.preprocess_settings["dir_out"])
            roi = self.preprocess_settings["roi"]
            if not isinstance(roi, (tuple, list)):
                roi = roi(real_path)
                print("\t\tRoi_res:{}, roi size: ({}, {})(w, h)".format(roi, roi[2] - roi[0], roi[3] - roi[1]))
            offset_x, offset_y = roi[0], roi[1]
            roi_w, roi_h = roi[2] - roi[0], roi[3] - roi[1]
            split_image(real_path, self.preprocess_settings["split"], out_dir=self.preprocess_settings["dir_out"],
                        roi=roi,
                        only_keep_defect_subimage=False)

            self.image_name_path_dict = dict()
            for image_file in for_each_image(self.preprocess_settings["dir_out"]):
                self.image_name_path_dict[os.path.basename(image_file)] = image_file

            # crop roi and save image
            ori_image = Image.open(real_path)
            roi_image = ori_image.crop(roi)
            dirname, filename = os.path.split(real_path)
            prefix_name, ext = os.path.splitext(filename)
            out_prefix = os.path.join(self.preprocess_settings["dir_out"], prefix_name)
            roi_image_path = "{}_roi{}".format(out_prefix, ext)
            print("\t\tRoi image saving to: {}".format(roi_image_path))
            roi_image.save(roi_image_path)

            # inference
            hor_index = []
            hor_score = []
            for i in range(self.split[0]):
                ver_index = []
                ver_score = []
                for j in range(self.split[1]):
                    print("\t\t\tInferencing {},{}".format(i, j))
                    pure_name, ext = os.path.splitext(os.path.basename(real_path))
                    sub_image_name = "{}_{}_{}_{}".format(pure_name, i, j, ext)
                    sub_image_path = self.image_name_path_dict[sub_image_name]
                    onnx_predict_index, onnx_predict_score = model_onnx.forward(sub_image_path)
                    ver_index.append(onnx_predict_index)
                    ver_score.append(onnx_predict_score)
                hor_index.append(np.concatenate(ver_index, axis=0))
                hor_score.append(np.concatenate(ver_score, axis=1))
            ori_index = np.concatenate(hor_index, axis=1)
            ori_score = np.concatenate(hor_score, axis=2)

            # defect filters
            print("Postprocess defect filter running!")
            out_name_postfix = []
            for postprocess in post_pipeline:
                print("\t{}".format(postprocess.__class__))
                ori_index, ori_score = postprocess.postprocess(ori_index, ori_score)
                out_name_postfix.append(postprocess.get_description())

            # paint res
            print("Do painting!")
            paint_res = None
            out_size = None
            for painter in self.painters:
                print("\t{}".format(painter.__class__))
                paint_res = painter.paint(ori_index, ori_score, paint_res, roi_image_path)
                if out_size is None:
                    out_size = painter.get_output_size()

            # save res image
            print("Saving image!")
            prefix_dir, filename = os.path.split(roi_image_path)
            filename, ext = os.path.splitext(filename)
            out_filename = "{}_res_{}{}".format(filename, "_".join(out_name_postfix), ext)
            out_path = os.path.join(self.preprocess_settings["dir_out"], out_filename)
            if not os.path.isdir(self.preprocess_settings["dir_out"]):
                os.makedirs(self.preprocess_settings["dir_out"])
            print(out_path)
            PainterBase.save_image(paint_res, out_path)

            # 标注文件导出到原图地址
            if os.path.isfile(roi_image_path):
                print("Transfer json annotation!")
                json_prefix, _ = os.path.splitext(real_path)
                out_json = "{}.json".format(json_prefix)
                roi_prefix, _ = os.path.splitext(roi_image_path)
                in_json = "{}.json".format(roi_prefix)
                swap_val = dict()
                swap_val["imagePath"] = os.path.split(real_path)[1]
                affine_labelme_annotation(in_json, out_json, (1, 1), (offset_x, offset_y), swap_value=swap_val)

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

if __name__ == "__main__":
    # kersen 项目配置
    kersen = Kersen()
    preprocess_settings = kersen.end2end_preprocess_setting()
    inference_settings, postprocess_settings = kersen.get_inference_and_postprocess_settings()
    e2e = End2End(preprocess_settings, inference_settings, postprocess_settings)
    e2e.run_end2end()
