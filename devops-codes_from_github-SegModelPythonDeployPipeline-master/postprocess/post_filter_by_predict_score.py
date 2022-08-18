from .post_filter_base import PostFilter


class PostFilterByPredictScore(PostFilter):

    def __init__(self, threshold_score_lower, threshold_score_higher=None, bg_index=0, ignore_index=None):
        """
        Args:
            threshold_score_lower:  预测分数超过该值才会被认为是特定类别，背景无效
            threshold_score_higher: TODO 未实现
            bg_index: background 索引
        """
        self.threshold_score_lower = threshold_score_lower
        self.threshold_score_higher = threshold_score_higher
        self.bg_index = bg_index
        self.ignore_index = ignore_index

    def get_description(self):
        return "Score"

    def postprocess(self, predict_label_map, predict_score_tensor):
        return self.post_filter_by_predict_score(predict_label_map, predict_score_tensor)

    def post_filter_by_predict_score(self, predict_label_map, predict_score_tensor):
        raise RuntimeError("need refine!!!")
        # local_predict_label_map = predict_label_map[0]
        # n, c, h, w = predict_score_tensor.shape
        # for channel_index in range(c):
        #     if channel_index == self.bg_index or (self.ignore_index and channel_index == self.ignore_index):
        #         continue
        #     score_map = np.copy(predict_score_tensor[0, channel_index, :, :])
        #
        #     original_non_current_defect_position = np.where(local_predict_label_map != channel_index)
        #     score_map[original_non_current_defect_position] = 1.
        #
        #     filter_target = np.where(score_map < self.threshold_score_lower[channel_index])
        #     local_predict_label_map[filter_target] = self.bg_index
        # return predict_label_map, predict_score_tensor
