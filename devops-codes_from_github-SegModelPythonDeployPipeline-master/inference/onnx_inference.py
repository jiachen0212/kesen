import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from utils import cv_imread_by_np


class OnnxInference:

    def __init__(self, model_path, input_size, mean, std, normalize_to_one=True, **kwargs):
        self.model_path = model_path
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.normalize_to_one = normalize_to_one

        self.onnx_session = ort.InferenceSession(model_path)

    def _read_image(self, path2image):
        image = cv_imread_by_np(path2image, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _resize_normalize(self, image):
        img = cv2.resize(image, tuple(self.input_size))
        img = np.array(img, dtype=np.float32)
        img = img[np.newaxis, :, :, :]  # n, h, w, c
        if self.normalize_to_one:
            img /= 255
        img -= np.float32(self.mean)
        img /= np.float32(self.std)
        img = np.transpose(img, [0, 3, 1, 2])  # n, h, w, c => n,c,h,w
        return img

    def forward(self, image):
        return self._forward_with_normalize(image)

    def _forward_with_normalize(self, image):
        if isinstance(image, str):  # if image is the path to the image
            image = self._read_image(image)
        image = self._resize_normalize(image)

        onnx_predict_logits = self._forward_naive(image)

        onnx_predict_score = softmax(onnx_predict_logits[0], 1)
        onnx_predict_index = np.argmax(onnx_predict_logits[0], axis=1)

        onnx_predict_score = onnx_predict_score[0, :, :, :]
        onnx_predict_index = onnx_predict_index[0, :, :]

        return onnx_predict_index, onnx_predict_score

    def _forward_naive(self, image_tensor):
        onnx_inputs = {self.onnx_session.get_inputs()[0].name: image_tensor.astype(np.float32)}
        onnx_predict = self.onnx_session.run(None, onnx_inputs)

        return onnx_predict
