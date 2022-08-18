import os

import numpy as np

from painter import *
from postprocess import *
from utils import cv_imread_by_np
from .setting_base import Settings


class KersenSide(Settings):
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
        # 侧面-彩色-拱形-面阵
        self.side_arch = {**self.preprocess_base,
                            "filters": [self.json_exist],
                            "roi": KersenSide.localize_arch_side,
                            "split": (2, 1)  #
                            }

        # 侧面-彩色-同轴-线扫
        self.side_coaxial = {**self.preprocess_base,
                            "filters": [self.json_exist],
                            "roi": KersenSide.localize_coaxial_side,
                            "split": (1, 4)  #
                            }

        # 侧面-彩色-同轴-线扫-长边
        self.side_coaxial_longer = {**self.preprocess_base,
                            "filters": [self.json_exist, self.filter_longer],
                            "roi": KersenSide.localize_coaxial_side,
                            "split": (1, 4)  #
                            }

        # 侧面-彩色-同轴-线扫-短边
        self.side_coaxial_shorter = {**self.preprocess_base,
                            "filters": [self.json_exist, self.filter_shorter],
                            "roi": KersenSide.localize_coaxial_side,
                            "split": (1, 3)  #
                            }

        # # 侧面-黑白-条纹-线扫
        # self.side_bar = {**self.preprocess_base,
        #                     "filters": [self.json_exist],
        #                     "roi": KersenSide.,
        #                     "split": (2, 1)  #
        #                     }

    def get_inference_and_postprocess_settings(self):
        feedback = self.arch_preprocess_settings()
        return feedback

    def get_preprocess_settings(self):
        return self.yao_preprocess_setting()

    def arch_preprocess_settings(self):
        #  分时明场
        bg_index = 0
        ignore_index = 5
        label_map = ["bg", "脏污异色", "脏污", "dds-dm", "damoyise-dm", "ignore"]
        inference_settings = {**self.inference_base,
                              "model_path": r"D:\Work\projects\KersenSide\deploy\大面\Bar\Deploy_Front_Surface_Bar_0623\11800_0622_e1.onnx",
                              "input_size": [1536, 2048],  # w, h
                              "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225],
                              "image_dir": r"D:\Work\projects\KersenSide\debug_out",
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
            "dir_out": r"D:\Work\projects\KersenSide\debug_out",
        }

        return inference_settings, post_process_settings

    def end2end_preprocess_setting(self):
        return {**self.side_arch,  # 配置OK 分时频闪
                "dir_in": r"D:\Work\projects\KersenSide\debug",
                "dir_out": r"D:\Work\projects\KersenSide\debug_out",
                "filters": [self.filter_ext_bmp],
                }

    def yao_preprocess_setting(self):
        settings = [
            {**self.side_arch,
             "dir_in": r"D:\Work\projects\kersen\data\side\侧边DDS功拱形光-建模",
             "dir_out": r"D:\Work\projects\kersen\data\side\debug\arch",
             },
            {**self.side_coaxial_longer,
             "dir_in": r"D:\Work\projects\kersen\data\side\脏污异色-同轴",
             "dir_out": r"D:\Work\projects\kersen\data\side\debug\coaxial",
             },
            {**self.side_coaxial_shorter,
             "dir_in": r"D:\Work\projects\kersen\data\side\脏污异色-同轴",
             "dir_out": r"D:\Work\projects\kersen\data\side\debug\coaxial",
             },
            {**self.side_coaxial_longer,
             "dir_in": r"D:\Work\projects\kersen\data\side\亮印-同轴",
             "dir_out": r"D:\Work\projects\kersen\data\side\debug\coaxial",
             },
            {**self.side_coaxial_shorter,
             "dir_in": r"D:\Work\projects\kersen\data\side\亮印-同轴",
             "dir_out": r"D:\Work\projects\kersen\data\side\debug\coaxial",
             },
        ]
        return settings

    # 侧面-彩色-拱形-面阵
    @staticmethod
    def localize_arch_edge(source_image, find_in_vertical=True, thre=None, expend=200):
        # timestamp_start = time.perf_counter()
        if len(source_image.shape) == 2:
            source_image = source_image[:, :, None]

        h, w, c = source_image.shape
        if find_in_vertical:  # ver
            sample_point = (int(w * 3 / 7), int(w * 1 / 2), int(w * 4 / 7))
            sample_lines = source_image[:, sample_point, :]
            mean_max = np.max(np.mean(sample_lines, 1), 1)
            if thre is None:
                thre = np.mean(mean_max) * 0.8
            low_bound_max = h
        else:  # hor
            sample_numbers = 7
            sample_point = [int(i * h / sample_numbers) for i in range(sample_numbers)]  # avoid center logo
            sample_lines = source_image[sample_point, :, :]
            mean_max = np.max(np.max(sample_lines, 0), 1)
            if thre is None:
                thre = np.mean(sample_lines)
            low_bound_max = w
        candidate = np.where(mean_max > thre)
        up_bound = candidate[0][0] - expend
        low_bound = candidate[0][-1] + expend
        up_bound = 0 if up_bound < 0 else up_bound
        low_bound = low_bound_max if low_bound > low_bound_max else low_bound

        # print(time.perf_counter() - timestamp_start)
        return up_bound, low_bound

    @staticmethod
    def localize_coaxial_edge(source_image, find_in_vertical=True, thre=None, expend=200):
        # timestamp_start = time.perf_counter()
        if len(source_image.shape) == 2:
            source_image = source_image[:, :, None]

        h, w, c = source_image.shape
        if find_in_vertical:  # ver
            sample_numbers = 15
            sample_point = [int(i * w / sample_numbers) for i in range(sample_numbers)]
            sample_lines = source_image[:, sample_point, :]
            mean_max = np.max(np.mean(sample_lines, 1), 1)
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
    def localize_arch_side(real_path_image, thre=None, expend=50):
        source_image = cv_imread_by_np(real_path_image)
        top, bottom = KersenSide.localize_arch_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
        left, right = KersenSide.localize_arch_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
        return left, top, right, bottom

    @staticmethod
    def localize_coaxial_side(real_path_image, thre=None, expend=50):
        source_image = cv_imread_by_np(real_path_image)
        top, bottom = KersenSide.localize_coaxial_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
        left, right = KersenSide.localize_coaxial_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
        return left, top, right, bottom

    @staticmethod
    def filter_ext_bmp(path_image):
        _, ext = os.path.splitext(path_image)
        if ext.lower() == ".bmp":
            return True
        return False

    @staticmethod
    def filter_longer(path_image):
        dir_img, filename = os.path.split(path_image)
        pure_name, ext = os.path.splitext(filename)
        name_splitted = pure_name.split("-")
        if len(name_splitted) != 3:
            print("Can't parse image name {}".format(path_image))
        else:
            if name_splitted[2] == "A" or name_splitted[2] == "C":
                return True
        return False

    @staticmethod
    def filter_shorter(path_image):
        dir_img, filename = os.path.split(path_image)
        pure_name, ext = os.path.splitext(filename)
        name_splitted = pure_name.split("-")
        if len(name_splitted) != 3:
            print("Can't parse image name {}".format(path_image))
        else:
            if name_splitted[2] == "B" or name_splitted[2] == "D":
                return True
        return False
