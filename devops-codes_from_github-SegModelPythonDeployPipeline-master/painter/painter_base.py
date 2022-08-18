import os
import random
from abc import abstractmethod

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import cv_imread_by_np, cv_imwrite


class PainterBase:

    def __init__(self, out_dir, label_map, bg_index=0, ignore_index=0, color_map=None, paint_in_one_image=False,
                 out_size_type="model"):
        self.out_dir = out_dir
        self.label_map = label_map
        self.paint_in_one_image = paint_in_one_image
        self.bg_index = bg_index
        self.ignore_index = ignore_index
        self.color_map = color_map if color_map else self.random_colors(label_map, bg_index, ignore_index)

        self.image_path = None
        self.original_size = None  # w,h
        self.out_size_type = out_size_type  # one of "model" or "ori"
        self.out_size = None

    @abstractmethod
    def _do_painting(self, onnx_predict_index, onnx_predict_score, candidate_image):
        """
        Args:
            onnx_predict_index: 模型预测的结果，每个位置是对应的判定index
            onnx_predict_score: nchw tensor 经过softmax
            candidate_image: 待绘制图像

        Returns:
            canvas_image: 返回绘制完毕的图像，可以接受None，不做任何返回
        """

    @staticmethod
    def save_image(image, path_out):
        cv_imwrite(image, path_out)

    @staticmethod
    def random_colors(label_map, bg_index, ignore_index):
        color_map = []
        for i in range(0, len(label_map)):
            if i == bg_index or i == ignore_index:
                color_map.append((0, 0, 0))
            else:
                color_map.append((random.randint(0, 255), random.randint(0, 255),
                                  random.randint(0, 255)))
        return color_map

    def _ensure_3_channal_image(self, candidate_image):
        if len(candidate_image.shape) == 2:
            candidate_image = np.expand_dims(candidate_image, 2)
        if candidate_image.shape[-1] == 1:
            candidate_image = np.repeat(candidate_image, 3, 2)
        return candidate_image

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_output_size(self, out_size):
        self.out_size = out_size  # w, h

    def get_output_size(self):
        return self.out_size  # w,h

    def _init_image(self):
        if not self.image_path:
            raise RuntimeError("{} image_path is not set.".format(self.__class__))
        feedback = cv_imread_by_np(self.image_path)
        if self.original_size is None:
            shape = feedback.shape
            h = shape[0]
            w = shape[1]
            self.original_size = (w, h)
        feedback = cv2.resize(feedback, self.out_size)
        return feedback

    def _get_font_path(self):
        current_path = os.path.realpath(__file__)
        path, filename = os.path.split(current_path)
        return os.path.join(path, "..", "resources", "simkai.ttf")

    def paint_text_with_Chinese(self, target_image, text, position, textSize, textColor):
        if (isinstance(target_image, np.ndarray)):
            target_image = Image.fromarray(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(target_image)

        font_path = self._get_font_path()
        fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
        trans_color = [textColor[2], textColor[1], textColor[0]]
        draw.text(position, text, tuple(trans_color), font=fontStyle)
        return cv2.cvtColor(np.asarray(target_image), cv2.COLOR_RGB2BGR)

    def paint(self, onnx_predict_index, onnx_predict_score, target_image, image_path=None):  # w, h
        if image_path:
            self.image_path = image_path
        if self.out_size is None:
            if len(onnx_predict_score.shape) == 4:
                onnx_predict_score = np.squeeze(onnx_predict_score, 0)
            elif len(onnx_predict_index.shape) == 3:
                onnx_predict_index = np.squeeze(onnx_predict_index, 0)

            if self.out_size_type == "model":  # 默认使用模型输出尺寸绘图
                c, h, w = onnx_predict_score.shape
                self.set_output_size((w, h))
            elif self.out_size_type == "ori":
                shape = cv_imread_by_np(self.image_path).shape
                h = shape[0]
                w = shape[1]
                self.set_output_size((w, h))
                onnx_predict_index = cv2.resize(onnx_predict_index, self.out_size, interpolation=cv2.INTER_NEAREST)

                onnx_predict_score = np.transpose(onnx_predict_score, (1, 2, 0))
                onnx_predict_score = cv2.resize(onnx_predict_score, self.out_size, interpolation=cv2.INTER_NEAREST)
                onnx_predict_score = np.transpose(onnx_predict_score, (2, 0, 1))

        if self.paint_in_one_image:
            canvas_image = target_image
            if canvas_image is None:
                canvas_image = self._init_image()
            tmp_res = self._do_painting(onnx_predict_index, onnx_predict_score, canvas_image)
            if tmp_res is not None:
                canvas_image = tmp_res
            return canvas_image
        else:
            canvas_image = self._init_image()
            tmp_res = self._do_painting(onnx_predict_index, onnx_predict_score, canvas_image)

            if tmp_res is not None:
                canvas_image = tmp_res
                if target_image is None:
                    target_image = canvas_image
                else:
                    target_image = np.concatenate((target_image, canvas_image), axis=1)
            return target_image
