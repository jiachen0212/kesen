
import json
import os
import sys
import cv2
import json
import numpy as np
cv2.CV_IO_MAX_IMAGE_PIXELS = 200224000


def train(image, five_holes_area, roi):
    m = cv2.matchTemplate(image, five_holes_area, cv2.TM_CCORR_NORMED)
    ys, xs = np.where(m == m.max())
    x, y = int(xs[0]), int(ys[0])
    area_points = [[x, y],
                   [x + five_holes_area.shape[1], y],
                   [x + five_holes_area.shape[1], y + five_holes_area.shape[0]],
                   [x, y + five_holes_area.shape[0]]]

    m = cv2.matchTemplate(image, roi, cv2.TM_CCORR_NORMED)
    ys, xs = np.where(m == m.max())
    left_top = (int(xs[0]), int(ys[0]))

    train_info = {
        # "roi": roi,
        "area_points": area_points,
        "mark_point": left_top,
        "score": float(m.max())
    }

    return train_info


def inference(image, train_info, verbose=False):
    roi = train_info.get("roi")
    area_points = train_info.get("area_points")
    area_points = np.array(area_points)
    mark_point = train_info.get("mark_point")

    m = cv2.matchTemplate(image, roi, cv2.TM_CCORR_NORMED)
    ys, xs = np.where(m == m.max())
    x, y = xs[0], ys[0]

    # 1. area_points - mark_point 新图像中的冗余在减法中被抵消;
    # 2. 新图像中template的坐标找到, 加到1.的结果上, 则得到完整五孔[area_points]的坐标啦.
    new_area_points = area_points - mark_point + (x, y)
    if verbose:
        draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        draw = cv2.drawContours(draw, [new_area_points], 0, (0, 255, 0), 2)
    else:
        draw = None
    inference_info = {
        "score": float(m.max()),
        "area_points": new_area_points.tolist(),
        "draw": draw
    }
    return inference_info


def main():
    if len(sys.argv) == 1:
        print("if you want to train, run command below:")
        print("python locate.py train [train_dir]\n")
        print("if you want to inference, run command below:")
        print("python locate.py inference [train_dir] [inference_list] [verbose: bool]")
        return

    train_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\train_dir'

    sig = sys.argv[1]
    if sig.lower().strip() == "train":
        # train_dir = sys.argv[2]
        
        img = cv2.imread(os.path.join(train_dir, "image.bmp"), 0)
        five_holes_area = cv2.imread(os.path.join(train_dir, "imac.bmp"), 0)
        roi = cv2.imread(os.path.join(train_dir, "template.bmp"), 0)
        train_info = train(img, five_holes_area, roi)
        with open(os.path.join(train_dir, "train_info.json"), "w") as f:
            json.dump(train_info, f, indent=4)

    if sig.lower().strip() == "inference":
        # train_dir = sys.argv[2]
        if len(sys.argv) >= 5:
            verbose = sys.argv[4].lower() == "true"
        else:
            verbose = False

        with open(os.path.join(train_dir, "train_info.json")) as f:
            train_info = json.load(f)

        roi = cv2.imread(os.path.join(train_dir, "template.bmp"), 0)
        train_info["roi"] = roi

        # inference_list = sys.argv[3]
        inference_list = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\codes\locate\infrence_dir\inference.txt'
        with open(inference_list) as f:
            for path in f:
                path = path.strip()
                if path.startswith("#") or not path:
                    continue
                img = cv2.imread(path, 0)
                print(path)
                inference_info = inference(img, train_info, verbose=verbose)
                draw = inference_info.pop("draw")

                draw_file ="draw.png"
                cv2.imwrite(draw_file, draw)

                json_file = ".".join(path.split(".")[:-1]) + ".json"
                with open(json_file, "w") as f1:
                    json.dump(inference_info, f1, indent=4)


if __name__ == '__main__':
    main()
