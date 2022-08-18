import json


def affine_labelme_annotation(anno_path, out_path, scale_factor, offset, swap_value=None):
    anno = None
    with open(anno_path, "r", encoding="utf-8") as reader:
        anno = json.load(reader)
    shapes = anno["shapes"]
    for shape in shapes:
        points = shape["points"]
        for point in points:
            point[0] = point[0] * scale_factor[0] + offset[0]
            point[1] = point[1] * scale_factor[1] + offset[1]

    if swap_value is not None:
        for key, val in swap_value.items():
            anno[key] = val
    with open(out_path, "w", encoding="utf-8") as writer:
        json.dump(anno, writer)
