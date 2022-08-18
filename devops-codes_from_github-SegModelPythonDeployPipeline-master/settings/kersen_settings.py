import os

import numpy as np

from painter import *
from postprocess import *
from utils import cv_imread_by_np
from .setting_base import Settings


class Kersen(Settings):

    def __init__(self):
        super().__init__()

        self.preprocess_base = {"defect": [],
                                "split": (1, 1),
                                "roi": (0, 0, 0, 0),
                                "filters": [self.json_exist],
                                "dir_in": r"",
                                "dir_out": None}

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
            "label_map": ["bg", "脏污异色", "ignore"],
            "painters": [],
            "dir_out": r""
        }

        """
        optical solutions
        """
        # 同轴
        # 8192*22000
        self.coaxial_base = {**self.preprocess_base,
                             "defect": ["disuanyise-dm", "ignore", "dds-dm-pengshang", "dds-dm-huashang",
                                        "liangyin-dm"],
                             "filters": [self.json_exist],
                             "roi": Kersen.localize_for_front_surface_Tunnel_Coaxial,
                             "split": (2, 4)}

        # 隧道
        # 8192*22000
        self.tunnel_base = {**self.preprocess_base,
                            "filters": [self.json_exist],
                            "defect": ["heixian", "fushidian", "huichen", "zangwu", "ignore", "znagwu"],
                            "roi": Kersen.localize_for_front_surface_Tunnel_Coaxial,
                            "split": (2, 4)}

        # 分时频闪
        self.bar_base = {**self.preprocess_base,
                         "defect": ["ignore", "dds-dm-pengshang", "dds-dm-huashang"],
                         "split": (4, 8),
                         "roi": Kersen.localize_for_front_surface_Bar,
                         }  # "roi": (0, 700, 16000, 40800)

        # 分时频闪-明场
        self.bar_bright_base = {**self.preprocess_base,
                                "defect": ["ignore", "zangwuyise"],
                                "split": (4, 4),
                                "roi": Kersen.localize_for_front_surface_Bar,
                                }  # "roi": (2000, 800, 14300, 20300)

    def get_inference_and_postprocess_settings(self):
        feedback = self.yao_bar()
        # feedback = self.yao_tunnel()
        # feedback = self.yao_coaxial()
        # feedback = self.jiachen_tunnel()
        return feedback

    def get_preprocess_settings(self):
        return self.yao_preprocess_setting()

    def yao_bar(self):
        #  分时明场
        bg_index = 0
        ignore_index = 5
        label_map = ["bg", "脏污异色", "脏污", "dds-dm", "damoyise-dm", "ignore"]
        inference_settings = {**self.inference_base,
                              "model_path": r"D:\Work\projects\kersen\deploy\大面\Bar\Deploy_Front_Surface_Bar_0623\11800_0622_e1.onnx",
                              "input_size": [1536, 2048],  # w, h
                              "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225],
                              "image_dir": r"D:\Work\projects\kersen\debug_out",
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
            "dir_out": r"D:\Work\projects\kersen\debug_out",
        }

        return inference_settings, post_process_settings

    def jiachen_tunnel(self):
        #  隧道
        bg_index = 0
        ignore_index = 5
        label_map = ["bg", "腐蚀点", "黑线", "脏污", "灰尘", "ignore"]
        inference_settings = {**self.inference_base,
                              "model_path": r"D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\station3_20220626_suidao_2000iter.onnx",
                              "input_size": [2000, 3000],
                              "mean": [123.675, 116.28, 103.53],
                              "std": [58.395, 57.12, 57.375],
                              "image_dir": r"D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir",
                              "label_map": label_map,
                              "bg_index": bg_index,
                              "scale_w_scale_h": [1.814, 1.628],
                              "roi": (0, 2100, 7256, 21640),
                              "split_target": (2, 4),
                              "H_full_W_full": [8192, 22000],
                              "normalize_to_one": False,
                              }

        post_process_settings = {**self.postprocess_base,
            # "post_pipeline": [
            #     PostFilterByPredictScore([0, 0.5, 0.5, 0.5, 0.5, 0.5], ignore_index=ignore_index),
            #     PostFilterByArea([0, 0, 0, 0, 0, 0], ignore_index=ignore_index),
            # ],
            "post_pipeline": {
                "PostFilterByPredictScore": [0, 0.5, 0.5, 0.5, 0.5, 0.5],
                "PostFilterByArea": [0, 0, 0, 0, 0, 0],
            },
            "bg_index": bg_index,
            "ignore_index": ignore_index,
            "label_map": label_map,
            "painters": [MaskPainter],
            "paint_in_one_image": False,
            "dir_out": r"D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\res_dir",
        }
        return inference_settings, post_process_settings

    def yao_coaxial(self):
        # 同轴
        bg_index = 0
        ignore_index = 5
        label_map = ["bg", "碰伤", "划伤", "滴酸异色", "亮印", "ignore"]
        inference_settings = {**self.inference_base,
                              "model_path": r"D:\Work\projects\kersen\deploy\0620\Coaxial_18400.onnx",
                              "input_size": [1536, 2048],  # w, h
                              "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225],
                              "image_dir": r"D:\Work\projects\kersen\test\vis_test_image\coaxial",
                              "label_map": label_map,
                              "bg_index": bg_index,
                              }
        post_process_settings = {**self.postprocess_base,
            "post_pipeline": [
                PostFilterByPredictScore([0, 0.5, 0.5, 0.5, 0.5, 0.5], ignore_index=ignore_index),
                PostFilterByArea([0, 0, 0, 0, 0, 0], ignore_index=ignore_index),
            ],
            "bg_index": bg_index,
            "ignore_index": ignore_index,
            "label_map": label_map,
            "painters": [MaskPainter],
            "paint_in_one_image": False,
            "dir_out": r"D:\Work\projects\kersen\test\vis_out_coaxial",
        }
        return inference_settings, post_process_settings

    def end2end_preprocess_setting(self):
        # # debug
        return {**self.bar_bright_base,  # 配置OK 分时频闪
                "dir_in": r"D:\Work\projects\kersen\debug",
                "dir_out": r"D:\Work\projects\kersen\debug_out",
                "filters": [self.filter_ext_bmp],
                }

    def yao_preprocess_setting(self):
        settings = [
            # {**self.tunnel_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0610_previous\黑线\银白色\隧道",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Tunnel\黑线_隧道",
            #  },
            # {**self.tunnel_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0610_previous\腐蚀点\银白色",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Tunnel\腐蚀点_隧道",
            #  },
            # {**self.coaxial_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0610_previous\滴酸异色大面",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Coaxial\滴酸异色大面_同轴",
            #  },
            # {**self.coaxial_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0610_previous\划伤\银白色\同轴",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Coaxial\划伤_同轴",
            #  },
            # {**self.coaxial_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0610_previous\亮印\同轴光",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Coaxial\亮印_同轴",
            #  },
            # {**self.coaxial_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0610_previous\碰伤\同轴",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Coaxial\碰伤_同轴",
            #  },
            # {**self.bar_bright_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0610_previous\脏污异色\脏污异色分时拆分后",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Bar\脏污异色_分时拆分",
            #  },
            #
            # # 0613 - 0614
            # {**self.bar_bright_base,  # 配置OK 分时频闪
            #  "dir_in": r"D:\Work\projects\kersen\data\0613-0614\0613-打磨异色【11】",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\0613-0614\Bar\打磨异色",
            #  "filters": [self.filter_ext_bmp,self.json_exist],
            #  },
            # {**self.bar_bright_base,  # 配置OK 分时频闪
            #  "dir_in": r"D:\Work\projects\kersen\data\0613-0614\0613-脏污异色【5】",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\0613-0614\Bar\脏污异色",
            #  "filters": [self.filter_ext_bmp,self.json_exist],
            #  },
            # {**self.bar_bright_base,  # 配置OK 分时频闪
            #  "dir_in": r"D:\Work\projects\kersen\data\0613-0614\0614 -DDS-废料",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\0613-0614\Bar\DDS-废料",
            #  "filters": [self.filter_ext_bmp,self.json_exist],
            #  },
            # # {**self.tunnel_base,  # 偏移严重  隧道
            # #  "dir_in": r"D:\Work\projects\kersen\data\0613-0614\0614-针眼-隧道【17】\隧道",
            # #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\0613-0614\Tunnel\针眼_隧道",
            # #  },
            # {**self.coaxial_base,  # 配置OK
            #  "dir_in": r"D:\Work\projects\kersen\data\0613-0614\0615亮印（9）同轴",
            #  "dir_out": r"D:\Work\projects\kersen\data\preprocess_output\Coaxial\0615亮印（9）同轴",
            #  },
        ]
        return settings

    @staticmethod
    def localize_one_edge(source_image, find_in_vertical=True, thre=None, expend=200):
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
            sample_point = (int(h * 2 / 7), int(h * 1 / 2), int(h * 5 / 7))  # avoid center logo
            sample_lines = source_image[sample_point, :, :]
            mean_max = np.max(np.max(sample_lines, 0), 1)
            if thre is None:
                thre = np.mean(mean_max)
            low_bound_max = w
        candidate = np.where(mean_max > thre)
        up_bound = candidate[0][0] - expend
        low_bound = candidate[0][-1] + expend
        up_bound = 0 if up_bound < 0 else up_bound
        low_bound = low_bound_max if low_bound > low_bound_max else low_bound

        # print(time.perf_counter() - timestamp_start)
        return up_bound, low_bound

    @staticmethod
    def localize_for_front_surface_Tunnel_Coaxial(real_path_image, thre=None, expend=200):
        source_image = cv_imread_by_np(real_path_image)
        top, bottom = Kersen.localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
        left, right = Kersen.localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
        return left, top, right, bottom

    @staticmethod
    def localize_for_front_surface_Bar(real_path_image, thre=None, expend=200):
        source_image = cv_imread_by_np(real_path_image)
        top, bottom = Kersen.localize_one_edge(source_image, find_in_vertical=True, thre=thre, expend=expend)
        left, right = Kersen.localize_one_edge(source_image, find_in_vertical=False, thre=thre, expend=expend)
        cal_bottom = top + 19500
        bottom = min(cal_bottom, bottom)
        return left, top, right, bottom

    @staticmethod
    def filter_middle_1(path_image):
        dir_img, filename = os.path.split(path_image)
        pure_name, ext = os.path.splitext(filename)
        name_splitted = pure_name.split("-")
        if len(name_splitted) != 3:
            print("Can't parse image name {}".format(path_image))
        else:
            if name_splitted[1] == "1":
                return True
        return False

    @staticmethod
    def filter_middle_2(path_image):
        dir_img, filename = os.path.split(path_image)
        pure_name, ext = os.path.splitext(filename)
        name_splitted = pure_name.split("-")
        if len(name_splitted) != 3:
            print("Can't parse image name {}".format(path_image))
        else:
            if name_splitted[1] == "2":
                return True
        return False

    @staticmethod
    def filter_ext_bmp(path_image):
        _, ext = os.path.splitext(path_image)
        if ext.lower() == ".bmp":
            return True
        return False
