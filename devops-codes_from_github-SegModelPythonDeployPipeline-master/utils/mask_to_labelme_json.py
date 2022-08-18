'''
  提出背景：
    有的类别模型的预测效果已经很不错了，可以直接套用预标注的结果。借助预标注，我们就可以大批量地对一些简单的数据进行标注来构建
    一个规模庞大的预训练数据集。这样可以显著地降低标注的耗时。代码改自晨皇。
'''
import copy
import json
import os

import cv2
import numpy as np
import scipy.special as sci


# 求内外环最短的距离pairs 输入分别为全部contours， 需要检查的index，亲属关系，disable
def find_nearest_point(contours, idx, relation, disable):
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


'''
example:
    MyJsonGenerator = MarkGenerator([0.7, 'json'], ['strip_patch', 'lumpy_patch'])
    第一个参数：[confidence阈值，json/mask，表示要转成json或者mask]
    第二个参数：缺陷列表，请注意不要填入'background'
    
    调用：
    MyJsonGenerator.run(predicition, ImgInfo, DirRoot)
    @prediction: Prediction after softmax
    @ImgInfo: 图片路径，用于生成json中的ImgPath字段
    @DirRoot: 目标Json要放在什么地方
'''


class MarkGenerator(object):
    __default_cfg__ = {
        "json_version": "4.2.5",
        "json_simple_factor": 1.5,
        "json_smallest_length": 20
    }

    def __init__(self, cfg, label_list):
        self.json_version = self.__default_cfg__['json_version']
        self.json_simple_factor = self.__default_cfg__['json_simple_factor']
        self.json_smallest_length = self.__default_cfg__['json_smallest_length']
        self.threshold = cfg[0]
        self.mode = cfg[1]
        self.label_list = label_list
        assert self.mode in ['json', 'mask']

    def run(self, prediction, pic_name, result_dir):
        if self.mode == 'json':
            self.run_json(prediction, pic_name, result_dir)
        else:
            raise NotImplementedError()

    def run_json(self, prediction, pic_name, result_dir):
        pesudo_mark = dict()
        pesudo_mark['version'] = self.json_version
        pesudo_mark['shapes'] = []
        prediction = sci.softmax(prediction, axis=1)
        prediction = prediction.squeeze(axis=0)

        for cl in range(1, len(prediction)):
            pred_mask = np.zeros((prediction.shape[1], prediction.shape[2])).astype(np.uint8)
            pred_mask[prediction[cl] > self.threshold] = 255

            pred_mask = cv2.medianBlur(pred_mask, 5)
            # print(np.unique(pred_mask, return_counts=True))
            contours, hierarchy = cv2.findContours(pred_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            # print(len(contours))
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
                    new_object['label'] = self.label_list[cl - 1]
                    new_object['group_id'] = None
                    new_object['shape_type'] = 'polygon'
                    new_object['flags'] = {}

                    ori_contour = copy.deepcopy(contours[i])

                    nearest_pair = find_nearest_point(contours, i, flags, disable)
                    if len(nearest_pair) != 0:
                        index_sorted = np.argsort(np.asarray(nearest_pair)[:, 2])
                        contours[i] = []
                        for k, idx in enumerate(index_sorted):
                            parent_index, child_index, parent_point_index, child_point_index = nearest_pair[idx]
                            if k == 0:
                                contours[i] += ori_contour[:parent_point_index]
                            contours[i] += [ori_contour[parent_point_index]]
                            child = contours[child_index][child_point_index:] + contours[child_index][
                                                                                :child_point_index + 1]
                            contours[i] += child
                            contours[i] += [ori_contour[parent_point_index]]
                            if k < len(index_sorted) - 1:
                                contours[i] += ori_contour[parent_point_index:nearest_pair[index_sorted[k + 1]][2]]
                            if k == len(index_sorted) - 1:
                                contours[i] += ori_contour[parent_point_index:]
                                contours[i] += [ori_contour[0]]

                    new_object["points"] = contours[i]
                    pesudo_mark['shapes'] += [new_object]

        debug_path = '{}/{}.json'.format(result_dir + '/annotation', pic_name[:-4])
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        if len(pesudo_mark['shapes']) > 0:
            pesudo_mark['imagePath'] = os.path.basename(pic_name)
            pesudo_mark['imageData'] = None
            pesudo_mark['imageHeight'] = prediction.shape[1]
            pesudo_mark['imageWidth'] = prediction.shape[2]
            with open(debug_path, 'w') as fp:
                json.dump(pesudo_mark, fp, indent=4)
