from abc import abstractmethod


class PostFilter:

    def get_description(self):
        """
        Returns: 当前过滤器的极简描述str，该str将被拼接到输出文件名中
        """
        return "Base"

    @abstractmethod
    def postprocess(self, predict_label_map, predict_score_tensor):
        """
        Args:
            predict_label_map: 每个位置标识当前像素预测为什么类型
            predict_score_tensor: 原始预测输出tensor

        Returns:
            predict_label_map: 该过滤器对结果过滤后的类别图
            predict_score_tensor: 原始预测输出tensor，无特殊原因不要修改
        """
