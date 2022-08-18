import copy
import json
import os

import cv2
import numpy as np

from utils import affine_labelme_annotation
from .painter_base import PainterBase


class LabelmeAnnoDumper(PainterBase):
    '''
    example:
        MyJsonGenerator = MarkGenerator([0.7, 'json'], ['strip_patch', 'lumpy_patch'])
        第一个参数：[confidence阈值，json/mask，表示要转成json或者mask]

        调用：
        MyJsonGenerator.run(predicition, ImgInfo, DirRoot)
        @prediction: Prediction after softmax
        @ImgInfo: 图片路径，用于生成json中的ImgPath字段
        @DirRoot: 目标Json要放在什么地方
    '''

    __default_cfg__ = {
        "json_version": "4.2.5",
        "json_simple_factor": 1.5,
        "json_smallest_length": 0
    }

    def __init__(self, out_dir, label_map, bg_index=0, ignore_index=0, color_map=None, paint_in_one_image=False,
                 out_size_type="model"):
        super().__init__(out_dir, label_map, bg_index=bg_index, ignore_index=ignore_index, color_map=color_map,
                         paint_in_one_image=paint_in_one_image, out_size_type=out_size_type)
        self.json_version = self.__default_cfg__['json_version']
        self.json_simple_factor = self.__default_cfg__['json_simple_factor']
        self.json_smallest_length = self.__default_cfg__['json_smallest_length']

        self.threshold = 0.5

    def _do_painting(self, onnx_predict_index, onnx_predict_score, candidate_image):
        """
        Args:
            onnx_predict_index: 模型预测的结果，每个位置是对应的判定index
            onnx_predict_score: nchw tensor 经过softmax
            candidate_image: 待绘制图像

        Returns:
            canvas_image: 返回绘制完毕的图像，可以接受None，不做任何返回
        """
        pseudo_mark = dict()
        pseudo_mark['version'] = self.json_version
        pseudo_mark['shapes'] = []

        for index in range(1, len(onnx_predict_score)):
            pred_mask = np.zeros((onnx_predict_score.shape[1], onnx_predict_score.shape[2])).astype(np.uint8)
            pred_mask[onnx_predict_score[index] > self.threshold] = 255

            contours, hierarchy = cv2.findContours(pred_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue
            disable = np.zeros(len(contours))
            flags = []

            for i in range(len(contours)):
                if cv2.arcLength(contours[i], True) < self.json_smallest_length:
                    disable[i] = 1
                if hierarchy[0][i][3] != -1:
                    flags += [hierarchy[0][i][3]]
                else:
                    flags += [-1]
            for i in range(len(contours)):
                contours[i] = cv2.approxPolyDP(contours[i], self.json_simple_factor, True)

            for i in range(len(contours)):
                contours[i] = contours[i].squeeze(1).tolist()
            for i in range(len(contours)):
                if flags[i] == -1 and disable[i] != 1:
                    new_object = dict()
                    new_object['label'] = self.label_map[index]
                    new_object['group_id'] = None
                    new_object['shape_type'] = 'polygon'
                    new_object['flags'] = {}

                    # ori_contour = copy.deepcopy(contours[i])
                    #
                    # nearest_pair = self.find_nearest_point(contours, i, flags, disable)
                    # if len(nearest_pair) != 0:
                    #     index_sorted = np.argsort(np.asarray(nearest_pair)[:, 2])
                    #     contours[i] = []
                    #     for k, idx in enumerate(index_sorted):
                    #         parent_index, child_index, parent_point_index, child_point_index = nearest_pair[idx]
                    #         if k == 0:
                    #             contours[i] += ori_contour[:parent_point_index]
                    #         contours[i] += [ori_contour[parent_point_index]]
                    #         child = contours[child_index][child_point_index:] + contours[child_index][
                    #                                                             :child_point_index + 1]
                    #         contours[i] += child
                    #         contours[i] += [ori_contour[parent_point_index]]
                    #         if k < len(index_sorted) - 1:
                    #             contours[i] += ori_contour[parent_point_index:nearest_pair[index_sorted[k + 1]][2]]
                    #         if k == len(index_sorted) - 1:
                    #             contours[i] += ori_contour[parent_point_index:]
                    #             contours[i] += [ori_contour[0]]

                    new_object["points"] = contours[i]
                    pseudo_mark['shapes'] += [new_object]

        debug_path = '{}.json'.format(self.image_path[:-4])
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        print(debug_path)
        if len(pseudo_mark['shapes']) > 0:
            pseudo_mark['imagePath'] = os.path.basename(self.image_path)
            pseudo_mark['imageData'] = None
            # pseudo_mark['imageHeight'] = onnx_predict_score.shape[1]
            # pseudo_mark['imageWidth'] = onnx_predict_score.shape[2]
            pseudo_mark['imageHeight'] = self.original_size[1]
            pseudo_mark['imageWidth'] = self.original_size[0]
            with open(debug_path, 'w', encoding="utf-8") as fp:
                json.dump(pseudo_mark, fp, indent=4)
            affine_labelme_annotation(debug_path, debug_path, (self.original_size[0] / onnx_predict_score.shape[2],
                                                               self.original_size[1] / onnx_predict_score.shape[1]),
                                      (0, 0))
        return None

    # 求内外环最短的距离pairs 输入分别为全部contours， 需要检查的index，亲属关系，disable
    def find_nearest_point(self, contours, idx, relation, disable):
        nearest_pair = []
        for i in range(len(contours)):
            distance = 1000000000
            temp_nearest_pair = []
            # 如果是children且可用：
            if relation[i] == idx and disable[i] == 0:
                for j in range(len(contours[idx])):
                    for k, child in enumerate(contours[i]):
                        temp_distance = (contours[idx][j][0] - contours[i][k][0]) ** 2 + (
                                contours[idx][j][1] - contours[i][k][1]) ** 2
                        # 更新distance和返回的index值
                        if temp_distance < distance:
                            distance = temp_distance
                            temp_nearest_pair = [idx, i, j, k]
            # 如果有返回值，append到结果里
            if len(temp_nearest_pair) == 4:
                nearest_pair += [temp_nearest_pair]
        return nearest_pair
