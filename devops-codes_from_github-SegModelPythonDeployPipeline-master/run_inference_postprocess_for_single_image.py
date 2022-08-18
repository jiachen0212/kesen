import os

from inference import OnnxInference
from painter import PainterBase
from settings import Kersen


class InferencePostprocess:
    def __init__(self, inference_settings, postprocess_settings):
        self.inference_settings = inference_settings
        self.postprocess_settings = postprocess_settings
        self.painters = []

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

    def run_inference_postprocess(self):
        model_onnx = OnnxInference(**self.inference_settings)
        image_dir = self.inference_settings["image_dir"]
        out_dir = self.postprocess_settings["dir_out"]
        post_pipeline = self.postprocess_settings["post_pipeline"]
        self.init_painters()
        for image_path in self.for_each_image(image_dir):
            # inference
            onnx_predict_index, onnx_predict_score = model_onnx.forward(image_path)
            # defect filters
            out_name_postfix = []
            for postprocess in post_pipeline:
                onnx_predict_index, onnx_predict_score = postprocess.postprocess(onnx_predict_index, onnx_predict_score)
                out_name_postfix.append(postprocess.get_description())
            # paint res
            paint_res = None
            for painter in self.painters:
                paint_res = painter.paint(onnx_predict_index, onnx_predict_score, paint_res, image_path)
            # save res image
            prefix_dir, filename = os.path.split(image_path)
            filename, ext = os.path.splitext(filename)
            out_filename = "{}{}{}".format(filename, "_".join(out_name_postfix), ext)
            out_path = os.path.join(out_dir, out_filename)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            print(out_path)
            PainterBase.save_image(paint_res, out_path)


if __name__ == "__main__":
    inference_settings, postprocess_settings = Kersen().get_inference_and_postprocess_settings()
    ip = InferencePostprocess(inference_settings, postprocess_settings)
    ip.run_inference_postprocess()
