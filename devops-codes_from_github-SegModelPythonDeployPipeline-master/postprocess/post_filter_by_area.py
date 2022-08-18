import cv2
import numpy as np

from .post_filter_base import PostFilter


class PostFilterByArea(PostFilter):

    def __init__(self, threshold_area_lower, threshold_area_higher=None, bg_index=0, ignore_index=None):
        """
        Args:
            threshold_area_lower: 面积大于此处设定数值将被保留，否则将被过滤，对背景无效
            threshold_area_higher: TODO 未实现 面积小于此处设定数值将被保留，否则将被过滤，对背景无效
            bg_index: background 索引
        """
        self.threshold_area_lower = threshold_area_lower
        self.threshold_area_higher = threshold_area_higher
        self.bg_index = bg_index
        self.ignore_index = ignore_index

    def get_description(self):
        return "Area"

    def postprocess(self, predict_label_map, predict_score_tensor):
        return self.post_filter_by_area(predict_label_map, predict_score_tensor)

    def post_filter_by_area(self, predict_label_map, predict_score_tensor):
        local_predict_label_map = predict_label_map[0]
        result_mask = np.ones(predict_label_map.shape)
        for each_label in range(len(self.threshold_area_lower)):
            if each_label == 0:
                continue
            if self.ignore_index and each_label == self.ignore_index:
                continue

            mask = np.array(local_predict_label_map == each_label, np.uint8)
            _, labels = cv2.connectedComponents(mask, 8)

            for label in np.unique(labels):
                if label == self.bg_index:
                    continue
                label_mask = labels == label
                area = np.sum(label_mask)
                if area < self.threshold_area_lower[each_label]:
                    result_mask[label_mask] = 0
        predict_label_map = predict_label_map * result_mask
        return predict_label_map, predict_score_tensor

# if __name__ == '__main__':
#     DebugImg = np.zeros((512, 512), dtype=np.uint8)
#     cv2.line(DebugImg, (10, 10), (10, 20), 1, 5)
#     cv2.rectangle(DebugImg, (50, 50), (60, 60), 1, 1)
#     DebugImgAfterPostPro = removeSmallDefect(DebugImg, [0, 100])
#     cv2.imwrite('/data/home/zhiquanli/myWork/PostProcessing/meta/general/debug_imgs/remove_short_defects_before.jpg',
#                 255 * DebugImg)
#     cv2.imwrite('/data/home/zhiquanli/myWork/PostProcessing/meta/general/debug_imgs/remove_short_defects_after.jpg',
#                 255 * DebugImgAfterPostPro)
