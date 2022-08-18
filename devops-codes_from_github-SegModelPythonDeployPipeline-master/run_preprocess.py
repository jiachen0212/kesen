import os

from preprocess import split_image
from settings import Kersen, Settings, KersenSide, KersenEdgeCorner

if __name__ == "__main__":
    # project = Kersen()  # kersen 大面
    # project = KersenSide()  # kersen 侧面、棱边
    project = KersenEdgeCorner()  # kersen 侧面、棱边

    settings = project.get_preprocess_settings()

    for setting in settings:
        print("Preprocessing {}".format(setting["dir_in"]))
        for real_path in Settings.filter_each_image(setting):
            print("\tProcessing {}".format(real_path))
            if not os.path.isdir(setting["dir_out"]):
                os.makedirs(setting["dir_out"])
            roi = setting["roi"]
            # TODO: 对一张图像内多个ROI区域兼容有点问题，暂时通过多次定位绕过
            if not isinstance(roi, (tuple, list)):
                roi = roi(real_path)
                print("\t\tRoi_res:{}, roi size: ({}, {})(w, h)".format(roi, roi[2] - roi[0], roi[3] - roi[1]))
            split_image(real_path, setting["split"], out_dir=setting["dir_out"], roi=roi,
                        only_keep_defect_subimage=False)
    print("DONE!")
