from .painter_base import PainterBase


class OriginalImagePainter(PainterBase):

    def _do_painting(self, onnx_predict_index, onnx_predict_score, candidate_image):
        feedback = self._init_image()
        feedback = self._ensure_3_channal_image(feedback)
        feedback = self.paint_text_with_Chinese(feedback, "原图", (10, 0), 50, (255, 255, 255))
        return feedback
