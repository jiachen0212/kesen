# coding=utf-8
import json
import os
import random


def phase_sampler(index_file_path, base_dir, phase_name, phase_ratio, out_file_ext=".txt"):
    """
    采样划分训练测试集合
    Args:
        index_file_path: 待划分的索引文件
        base_dir: 输出索引文件的目录地址
        phase_name: 划分的阶段名称，通常["train", "eval"]
        phase_ratio: 划分的阶段数据占比，和phase_name一一映射，通常[7,3]
        out_file_ext: 输出索引文件后缀，默认".txt"

    Returns:

    """
    all_items = []
    with open(index_file_path, "r", encoding="utf-8") as reader:
        for line in reader:
            all_items.append(line)

    totoal_sample_num = len(all_items)
    ratio_sum = float(sum(phase_ratio))
    phase_ratio = [i / ratio_sum for i in phase_ratio]
    phase_sample_num = [int(totoal_sample_num * ratio) for ratio in phase_ratio[:-1]]
    phase_sample_num.append(totoal_sample_num - sum(phase_sample_num))

    random.shuffle(all_items)

    last_index = 0
    for i, name in enumerate(phase_name):
        start = last_index
        end = start + phase_sample_num[i]
        last_index = end
        samples_in_current_phase = all_items[start:end]
        name = name + out_file_ext
        with open(os.path.join(base_dir, name), "w") as writer:
            writer.writelines(samples_in_current_phase)


def generate_index(path_base, anno_ext=".json", only_image_with_defect=False):
    """
    生成 Smore_seg 可用 index 文件

    Args:
        path_base: 目标文件夹文件夹会被递归查找，所有能配对的图像和标注会被记录；图像接受的后缀('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        anno_ext: 标注文件后缀，匹配用。默认".json"
        only_image_with_defect: 没有标注的图像将不会被写到索引文件中，ignore将被忽略，如果一个图像内所有标注都为ignore类型，该图像仍被视为无有效标注图像。

    Returns:
        1. 所有经过筛选有效的[图像，标注]对记录文件 all_samples.txt
        2. 所有缺陷 *实例* 个数统计 info.json
        3. 缺陷-所在图像的倒排索引
    """
    out_file = os.path.join(path_base, "all_samples.txt")
    record = dict()
    record["path_dataset"] = path_base
    with open(out_file, "w", encoding="utf-8") as writer:
        img_ext = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        for root, dirs, files in os.walk(path_base):
            for filename in files:
                filename_base, ext = os.path.splitext(filename)
                if ext in img_ext:
                    path_target_annotation = os.path.join(root, filename_base + anno_ext)
                    if os.path.isfile(path_target_annotation):
                        source = str(os.path.join(root, filename))
                        target = str(path_target_annotation)
                        oneline = source + "," + target + "\n"
                        if only_image_with_defect:
                            if parse_annotation(path_target_annotation, record, oneline):
                                writer.write(oneline)
                        else:
                            writer.write(oneline)

    pop_keys = []
    for key, val in record.items():
        if isinstance(val, set):
            image_index = list(val)
            with open(os.path.join(path_base, "{}_image_index.txt".format(key)), "w", encoding="utf-8") as writer:
                for line in image_index:
                    writer.write(line)
            pop_keys.append(key)
    for key in pop_keys:
        record.pop(key)
    with open(os.path.join(path_base, "info.json"), "w", encoding="utf-8") as writer:
        json.dump(record, writer, ensure_ascii=False, indent=4)
    return out_file


def get_labels(path_annotation):
    feedback = []
    with open(path_annotation, "r", encoding="utf-8") as reader:
        json_obj = json.load(reader)
        annotations = json_obj["shapes"]
        for anno in annotations:
            category = anno["label"]
            feedback.append(category)
    return feedback


def parse_annotation(path_target_annotation, record, record_line):
    categories = record.get("categories", dict())
    new_labels = get_labels(path_target_annotation)
    feedback = False
    for label in new_labels:
        if label != "ignore":
            feedback = True
        categories[label] = categories.get(label, 0) + 1
        image_have_currentlable = record.get(label, set())
        image_have_currentlable.add(record_line)
        record[label] = image_have_currentlable
    record["categories"] = categories
    return feedback


if __name__ == "__main__":
    path_list = [r"/data/home/hengzhiyao/codes/SMore-Seg/newdata_Data/kersen/all_before_0611/分时明场",
                 r"/data/home/hengzhiyao/codes/SMore-Seg/newdata_Data/kersen/all_before_0611/隧道",
                 r"/data/home/hengzhiyao/codes/SMore-Seg/newdata_Data/kersen/all_before_0611/同轴"]
    for path_base in path_list:
        phase_name = ["train", "eval"]
        phase_ratio = [9, 1]
        path_generated_index_file = generate_index(path_base, only_image_with_defect=False)
        phase_sampler(path_generated_index_file, path_base, phase_name, phase_ratio)
