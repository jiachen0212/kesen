import os

import numpy as np

from painter import *
from postprocess import *
from utils import cv_imread_by_np
from .setting_base import Settings


class KersenEdgeCorner(Settings):
    def __init__(self):
        super().__init__()

        self.preprocess_base = {"defect": [],
                                "split": (1, 1),
                                "roi": (0, 0, 0, 0),
                                "filters": [self.json_exist],
                                "dir_in": r"",
                                "dir_out": None
                                }

        self.inference_base = {
            "model_type": "onnx",
            "model_path": r"",
            "input_size": [1024, 1024],  # w, h
            "mean": [],
            "std": [],
            "image_dir": r"",
            "label_map": [],
            "normalize_to_one": True
        }

        self.postprocess_base = {
            "post_pipeline": [
                PostFilterByPredictScore
            ],
            "out_size_type": "model",  # one of "ori" or "model"
            "label_map": [],
            "painters": [],
            "dir_out": r""
        }

        """
        optical solutions
        """
        # 棱边-彩色-？？-面阵-下物料边缘
        self.edge_lower = {**self.preprocess_base,
                            "filters": [self.json_exist],
                            "roi": KersenEdgeCorner.localize_for_lower_piece_edge,
                            "split": (1, 1)  #
                            }

        # 棱边-彩色-？？-面阵-上物料边缘
        self.edge_upper = {**self.preprocess_base,
                            "filters": [self.json_exist],
                            "roi": KersenEdgeCorner.localize_for_upper_piece_edge,
                            "split": (1, 1)  #
                            }

        # R角 - 彩色 -？？-面阵
        self.corner = {**self.preprocess_base,
                            "filters": [self.json_exist],
                            "roi": KersenEdgeCorner.localize_for_corner,
                            "split": (1, 1)  #
                            }

    def get_inference_and_postprocess_settings(self):
        feedback = self.edge()
        return feedback

    def get_preprocess_settings(self):
        return self.preprocess_setting()

    def edge(self):
        #  分时明场
        bg_index = 0
        ignore_index = 5
        label_map = ["bg", "脏污异色", "脏污", "dds-dm", "damoyise-dm", "ignore"]
        inference_settings = {**self.inference_base,
                              "model_path": r"D:\Work\projects\Kersen\deploy\大面\Bar\Deploy_Front_Surface_Bar_0623\11800_0622_e1.onnx",
                              "input_size": [1536, 2048],  # w, h
                              "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225],
                              "image_dir": r"D:\Work\projects\Kersen\debug_out",
                              "label_map": label_map,
                              "bg_index": bg_index,
                              }
        post_process_settings = {**self.postprocess_base,
            "post_pipeline": [
                # PostFilterByPredictScore([0, 0.5, 0.5], ignore_index=ignore_index),
                # PostFilterByArea([0, 0, 0], ignore_index=ignore_index),
                # PostFilterByRectHeightWidth([0, 0, 0], rotate_rect=True, ignore_index=ignore_index),
            ],
            "bg_index": bg_index,
            "ignore_index": ignore_index,
            "label_map": label_map,
            "out_size_type": "model",  # one of "ori" or "model"
            "painters": [MaskPainter, LabelmeAnnoDumper],
            "paint_in_one_image": False,
            "dir_out": r"D:\Work\projects\Kersen\debug_out",
        }

        return inference_settings, post_process_settings

    def end2end_preprocess_setting(self):
        return {**self.edge_upper,  # 配置OK 分时频闪
                "dir_in": r"D:\Work\projects\Kersen\debug",
                "dir_out": r"D:\Work\projects\Kersen\debug_out",
                "filters": [self.filter_ext_bmp],
                }

    def preprocess_setting(self):
        settings = [
            # {**self.edge_lower,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\side\侧边DDS棱边-建模",
            #  "dir_out": r"D:\Work\projects\Kersen\data\side\debug\edge_lower",
            #  },
            # {**self.edge_upper,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\side\侧边DDS棱边-建模",
            #  "dir_out": r"D:\Work\projects\Kersen\data\side\debug\edge_heigher",
            #  },
            {**self.corner,  # 配置OK
             "dir_in": r"D:\Work\projects\kersen\data\side\DDS-R角",
             "dir_out": r"D:\Work\projects\Kersen\data\side\debug\corner",
             },
        ]
        return settings

    @staticmethod
    def localize_corner_edge(source_image, find_in_vertical=True, thre=None, expend=200):
        # timestamp_start = time.perf_counter()
        if len(source_image.shape) == 2:
            source_image = source_image[:, :, None]

        h, w, c = source_image.shape
        if find_in_vertical:  # ver
            sample_numbers = 3
            sample_point = [int((i + 1) * w / (sample_numbers + 2)) for i in range(sample_numbers)]
            sample_lines = source_image[:, sample_point, :]
            mean_max = np.max(np.max(sample_lines, 1), 1)
            if thre is None:
                thre = np.mean(mean_max) * 0.8
            low_bound_max = h
        else:  # hor
            sample_numbers = 7
            sample_point = [int(i * h / sample_numbers) for i in range(sample_numbers)]
            sample_lines = source_image[sample_point, :, :]
            mean_max = np.max(np.max(sample_lines, 0), 1)
            if thre is None:
                thre = np.mean(sample_lines) * 3
            low_bound_max = w
        candidate = np.where(mean_max > thre)
        up_bound = candidate[0][0] - expend
        low_bound = candidate[0][-1] + expend
        up_bound = 0 if up_bound < 0 else up_bound
        low_bound = low_bound_max if low_bound > low_bound_max else low_bound

        # print(time.perf_counter() - timestamp_start)
        return up_bound, low_bound

    @staticmethod
    def localize_for_corner(real_path_image, thre=None, expend=100):
        source_image = cv_imread_by_np(real_path_image)
        top, bottom = KersenEdgeCorner.localize_corner_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
        # left, right = localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
        left = 500
        right = 1900
        return left, top, right, bottom

    @staticmethod
    def localize_one_edge(source_image, find_in_vertical=True, thre=None, expend=200):
        # timestamp_start = time.perf_counter()
        if len(source_image.shape) == 2:
            source_image = source_image[:, :, None]

        h, w, c = source_image.shape
        if find_in_vertical:  # ver
            sample_numbers = 5
            sample_point = [int(i * w / sample_numbers) for i in range(sample_numbers)]
            sample_lines = source_image[:, sample_point, :]
            mean_max = np.max(np.mean(sample_lines, 1), 1)
            if thre is None:
                thre = np.mean(mean_max)
            low_bound_max = h
        else:  # hor
            sample_numbers = 9
            sample_point = [int(i * h / (sample_numbers + 2)) for i in range(sample_numbers)]
            sample_lines = source_image[sample_point, :, :]
            mean_max = np.max(np.mean(sample_lines, 0), 1)
            if thre is None:
                thre = np.mean(mean_max) * 0.5
            low_bound_max = w
        candidate = np.where(mean_max > thre)
        up_bound = candidate[0][0] - expend
        low_bound = candidate[0][-1] + expend
        up_bound = 0 if up_bound < 0 else up_bound
        low_bound = low_bound_max if low_bound > low_bound_max else low_bound

        # print(time.perf_counter() - timestamp_start)
        return up_bound, low_bound

    @staticmethod
    def localize_for_upper_piece_edge(real_path_image, thre=None, expend=200):
        source_image = cv_imread_by_np(real_path_image)
        top, bottom = KersenEdgeCorner.localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=0)
        left, right = KersenEdgeCorner.localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=0)

        upper_piece_top = top + 150
        upper_piece_bottom = top + 300
        return left, upper_piece_top, right, upper_piece_bottom

    @staticmethod
    def localize_for_lower_piece_edge(real_path_image, thre=None, expend=200):
        source_image = cv_imread_by_np(real_path_image)
        top, bottom = KersenEdgeCorner.localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=0)
        left, right = KersenEdgeCorner.localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=0)

        lower_piece_top = bottom - 300
        lower_piece_bottom = bottom - 150
        return left, lower_piece_top, right, lower_piece_bottom

    @staticmethod
    def filter_ext_bmp(path_image):
        _, ext = os.path.splitext(path_image)
        if ext.lower() == ".bmp":
            return True
        return False
