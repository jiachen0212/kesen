import os
from abc import abstractmethod


class Settings:
    def __init__(self):
        self.preprocess_base = {
            "defect": [],
            "split": (1, 1),
            "roi": (0, 0, 0, 0),
            "filters": [],
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
            "post_pipeline": [],
            "label_map": [],
            "bg_index": 0,
            "ignore_index": -1,
            "painters": [],
            "paint_in_one_image": False,
            "dir_out": r""
        }

    @abstractmethod
    def get_inference_and_postprocess_settings(self):
        """
        Returns: 根据实际情况配置 self.inference_base, self.postprocess_base 后返回

        return self.inference_base, self.postprocess_base
        """
        raise NotImplementedError("{} get_inference_and_postprocess_settings not implemented!")

    @abstractmethod
    def get_preprocess_settings(self):
        """
        Returns: 根据实际情况配置 self.preprocess_base 返回

        return self.preprocess_base
        """

    @staticmethod
    def filter_each_image(setting):
        src = setting["dir_in"]
        filters = setting["filters"]

        for root, dirs, files in os.walk(src):
            for file in files:
                abs_file_path = os.path.join(root, file)
                do_preprocess = True
                if filters:
                    for filter in filters:
                        if not filter(abs_file_path):
                            do_preprocess = False
                            break
                if do_preprocess:
                    yield abs_file_path

    @staticmethod
    def json_exist(path_image):
        filename, ext = os.path.splitext(path_image)
        if ext not in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
            return False
        json_path = filename + ".json"
        if os.path.isfile(json_path):
            return True
        return False
