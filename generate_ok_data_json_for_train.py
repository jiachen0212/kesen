import json
import os
'''
抑制过杀用的, 完全ok无任何缺陷的数据生成一个空json-labelme
告诉模型非缺陷样子是啥.[抑制过杀用]

'''
from PIL import Image


def generate_img_index(root_dir, extra_function=None, check_json_exist=False):
    out_filename = r"all_img_index.txt"
    with open(os.path.join(root_dir, out_filename), "w", encoding="utf-8") as writer:
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                pure_filename, ext = os.path.splitext(filename)
                if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
                    if check_json_exist:
                        json_filename = pure_filename + ".json"
                        if not os.path.isfile(os.path.join(root, json_filename)):
                            continue
                    if extra_function:
                        extra_function(os.path.join(root, filename), writer)
                    else:
                        writer.write("{}\n".format(os.path.join(root, filename)))


def extra_empty_labelme_annotation(image_path, index_writer):
    """
    {
      "version": "4.6.0",
      "flags": {},
      "shapes": null,
      "imagePath": "",
      "imageData": null,
      "imageHeight": 8000,
      "imageWidth": 8192
    }
    """
    out_file_prefix, img_ext = os.path.splitext(image_path)
    _, image_filename = os.path.split(image_path)

    out_labelme_filepath = "{}.json".format(out_file_prefix)

    img = Image.open(image_path)
    w, h = img.size

    json_obj = dict()
    json_obj["version"] = "4.6.0"
    json_obj["flags"] = {}
    json_obj["shapes"] = []
    json_obj["imagePath"] = image_filename
    json_obj["imageData"] = None
    json_obj["imageHeight"] = h
    json_obj["imageWidth"] = w

    if not os.path.isfile(out_labelme_filepath):
        with open(out_labelme_filepath, "w", encoding="utf-8") as writer:
            json.dump(json_obj, writer)
    else:
        print("{} exist!".format(out_labelme_filepath))

    index_writer.write("{},{}\n".format(image_path, out_labelme_filepath))


if __name__ == "__main__":
    # root_dir = r"D:\Work\projects\hebi\石墨面\station2_data\0408\20220408 PF分切线气泡 Q情况 工位2存在漏检\input_image_data\2"
    # generate_img_index(root_dir, extra_empty_labelme_annotation)

    # base_dir = r"D:\Work\projects\hebi\psa_station4\20220423"

    # root_dir = os.path.join(base_dir,r"1")
    # generate_img_index(root_dir)
    # root_dir = os.path.join(base_dir,r"2")
    # generate_img_index(root_dir)

    # root_dir = os.path.join(base_dir,r"4")
    # generate_img_index(root_dir)

    # generate_img_index(r"D:\Work\projects\新吴光电\data\20220528采集缺陷")
    # generate_img_index(r"D:\Work\projects\hebi\shimo\station2_data\0721", check_json_exist=True)

    generate_img_index(r"D:\BaiduNetdiskDownload\out_0805\8_out", extra_empty_labelme_annotation)
