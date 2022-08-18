import cv2
import numpy as np

from .painter_base import PainterBase


class MaskPainter(PainterBase):

    def gen_mask(self, segm):
        mask = np.zeros((segm.shape[0], segm.shape[1], 3), dtype=np.uint8)
        for i in range(0, len(self.label_map)):
            target = np.where(segm == i)
            mask[target[0], target[1], :] = self.color_map[i]
        return mask

    def _do_painting(self, onnx_predict_index, onnx_predict_score, candidate_image):
        candidate_image = self._ensure_3_channal_image(candidate_image)

        mask = self.gen_mask(onnx_predict_index)

        mask_weight = 0.3
        candidate_image = cv2.addWeighted(candidate_image, 1 - mask_weight, mask, mask_weight, 0)

        for i in range(0, len(self.label_map)):
            if i == self.ignore_index or i == self.bg_index:
                text_color = [255, 255, 255]
            else:
                text_color = self.color_map[i]
            candidate_image = self.paint_text_with_Chinese(candidate_image, self.label_map[i], (10, 50 * i), 50,
                                                           text_color)
        return candidate_image
