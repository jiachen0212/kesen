import logging

import cv2
import numpy as np
from config import general_config
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import softmax
from misc.contour_resize import resize_contour
import time


class PostProcess:
    """
    predict: 1 x H x W
    score_map: B x C x H x W 需要经过softmax

    """

    def __init__(self, predict, score_map, general_config, soft_max=False) -> None:
        if score_map is not None:
            assert len(score_map.shape) == 4  # B, C, H, W
            self.score_map = softmax(score_map, axis=1) if soft_max else score_map
            self.predict = np.argmax(self.score_map, 1)[0, :, :]
        else:
            self.predict = predict

        self.config = general_config
        self.numb_classes = self.config['numb_classes']
        self.area_remove_thresh = self.config['defect_info'].get('area_remove_thresh', None)
        self.bd = self.config['defect_info'].get('boundary_filter', None)
        self.mbd = self.config['defect_info'].get('min_boundary_filter', None)
        self.pro = self.config['defect_info'].get('prob_thre', None)
        self.contour_resize_on_filter = self.config['defect_info'].get('contour_resize_on_filter', None)
        self.contour_resize_on_pred = self.config.get('pred_resize_info', None)
        if self.pro is not None:
            self.heat_map = np.max(self.score_map[0, :, :, :], axis=0)

    def info_check(self):

        input_size = self.config['input_info']['size']
        if input_size:
            assert input_size == self.predict[:2]
        if self.area_remove_thresh is not None:
            assert self.numb_classes == len(self.area_remove_thresh)
        if self.contour_resize_on_filter is not None:
            assert self.contour_resize_on_filter['type'] in \
                   ['by_centroid', 'equidistant_enlarge', 'proportional_enlarge'], NotImplementedError
        if self.contour_resize_on_pred is not None:
            op_type = self.contour_resize_on_pred.get('type')
            assert op_type in ['by_centroid', 'equidistant_enlarge', 'proportional_enlarge'], NotImplementedError

    def forward(self):
        self.info_check()

        # determine which prediction would be preserved.
        result_mask = np.copy(self.predict)

        for each_label in range(self.numb_classes):
            result_info = {'number_c': self.numb_classes}
            # skip the background.
            if each_label == 0:
                continue
            # mask for each label
            mask = np.array(self.predict == each_label, np.uint8)
            # 使用connectedComponentsWithStats能够直接输出面积和boundingbox
            cc_output = cv2.connectedComponentsWithStats(mask, 8)
            num_labels  = cc_output[0]
            cc_stats = cc_output[2]
            cc_labels = cc_output[1]

            # instance for each label
            for label in range(num_labels):
                if label == 0:
                    continue
                
                x = cc_stats[label, cv2.CC_STAT_LEFT]
                y = cc_stats[label, cv2.CC_STAT_TOP]
                w = cc_stats[label, cv2.CC_STAT_WIDTH]
                h = cc_stats[label, cv2.CC_STAT_HEIGHT]
                area = cc_stats[label, cv2.CC_STAT_AREA]

                # 原来是 label_mask = labels == label 对每一个component进行全图拷贝低效
                # 改成shallow slicing对局部进行操作即可，剩下的findContour也不需要全图操作而是局部操作
                roi_mask = cc_labels[y:y+h, x:x+w]
                result_mask_out_temp = result_mask[y:y+h, x:x+w]
                label_mask = roi_mask == label
                result_info['bd'] = [1]

                if self.area_remove_thresh is not None:
                    result_info['area'] = area
                if self.mbd is not None or\
                    self.contour_resize_on_filter is not None or\
                    self.contour_resize_on_pred is not None:
                    # 仅当需要计算minbox和contour时需要findContours
                    contours, _ = cv2.findContours(label_mask.astype(np.uint8), cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE, offset=(x, y))
                    contours = list(contours)
                    if self.contour_resize_on_filter is not None:
                        self.predict, contours = resize_contour(self.predict, each_label, False, contours, self.contour_resize_on_filter)
                    if self.contour_resize_on_pred is not None:
                        self.predict, contours = resize_contour(self.predict, each_label, True, contours, self.contour_resize_on_pred)
                    contours = tuple(contours)
                    result_info['bd'] = contours

                if self.pro is not None:
                    roi_score = label_mask * self.heat_map[y:y+h, x:x+w]
                    # roi_score = label_mask * self.heat_map
                    mean_value_of_roi = np.sum(roi_score) / np.sum(label_mask)
                    if self.pro['mean'] is not None:
                        result_info['meanValueOfRoi'] = mean_value_of_roi

                    if self.pro['var'] is not None:
                        var_map = roi_score - mean_value_of_roi
                        var_map[label_mask == 0] = 0
                        var_map = var_map ** 2
                        std = np.sum(var_map) / (np.sum(label_mask))
                        result_info['stdValueOfRoi'] = std

                    if self.pro['entropy'] is not None:
                        assert self.score_map.shape[0] == 1, "we only support this function when batch_size==1"
                        assert np.unique(self.score_map)[0] >= 0 and np.unique(self.score_map)[
                            -1] <= 1, "the score_map is not valid. " \
                                      "please check whether you've softmax your feature map."
                        entropy_per_px = stats.entropy(self.score_map, axis=1)  # class axis
                        entropy_per_px = entropy_per_px * label_mask
                        avg_entropy = np.sum(entropy_per_px) / np.sum(label_mask)
                        result_info['avgEntropy'] = avg_entropy

                if 'area' in result_info.keys():
                    if result_info['area'] is not None:
                        if result_info['area'] < self.area_remove_thresh[each_label]:
                            result_mask_out_temp[label_mask] = 0
                            continue # 当前component过滤后可以continue，不需要剩下的操作
                if 'bd' in result_info.keys():
                    if self.bd is not None:
                        if self.bd['height'] is not None:
                            if h < self.bd['height'][each_label]:
                                result_mask_out_temp[label_mask] = 0
                                continue
                        if self.bd['width'] is not None:
                            if w < self.bd['width'][each_label]:
                                result_mask_out_temp[label_mask] = 0
                                continue
                    if result_info['bd'] is not None and self.mbd is not None:
                        assert len(result_info['bd']) == 1
                        rect = cv2.minAreaRect(result_info['bd'][0])
                        w, h = rect[1]
                        if self.mbd['height'] is not None:
                            if h < self.mbd['height'][each_label]:
                                result_mask_out_temp[label_mask] = 0
                                continue
                        if self.mbd['width'] is not None:
                            if w < self.mbd['width'][each_label]:
                                result_mask_out_temp[label_mask] = 0
                                continue
                    if result_info['bd'] is not None and self.contour_resize_on_filter is not None:
                        thickness = self.contour_resize_on_filter.get('thickness', 3)
                        if isinstance(thickness, list):
                            thickness = thickness[each_label]
                        result_mask = cv2.drawContours(result_mask, result_info['bd'], 0, 1, thickness)
                if self.pro is not None:
                    if self.pro['mean'] is not None:
                        if result_info['meanValueOfRoi'] < self.pro['mean']:
                            result_mask_out_temp[label_mask] = 0
                            print("an instance of {} is omitted due to its mean {} is lower "
                                         "than the threshold".format(each_label, result_info['meanValueOfRoi']))
                    if self.pro['var'] is not None:
                        if result_info['stdValueOfRoi'] > self.pro['var']:
                            result_mask_out_temp[label_mask] = 0
                            print("an instance of {} is omitted due to its var {} is larger "
                                         "than the threshold".format(each_label, result_info['stdValueOfRoi']))
                    if self.pro['entropy'] is not None:
                        if result_info['avgEntropy'] > self.pro['entropy']:
                            result_mask_out_temp[label_mask] = 0
                            print("an instance of {} is omitted due to its entropy {} is larger "
                                         "than the threshold".format(each_label, result_info['avgEntropy']))
        pred = self.predict * result_mask
        return pred


if __name__ == '__main__':
    predict = cv2.imread('/Users/zhiquanli/Documents/smartmore/工具/meta/debug/binary_image.bmp', 0)
    # score map 测试：自由调整所有前景和背景的置信度
    # score_map = np.zeros((1, 2, predict.shape[0], predict.shape[1]))
    # score_map[0][1][predict == 255] = 0.7
    # score_map[0][0][predict == 255] = 0.3
    # score_map[0][0][predict != 255] = 1.0
    predict[predict == 255] = 1 # 255改成1

    assert predict is not None, 'predict image is None.'

    Sdk = PostProcess(predict, None, general_config)

    t1 = time.time()
    for i in range(10):
        filer_re = Sdk.forward()
    t2 = time.time()

    print("running time: ", (t2-t1) / 10)


    cv2.imwrite('/Users/zhiquanli/Documents/smartmore/工具/meta/debug/output.bmp', filer_re*255) # 1改成255进行可视化
