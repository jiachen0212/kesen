import argparse
import base64
import json
import os
import os.path as osp
import warnings
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import yaml

from labelme import utils


def main(json_file, labelme_vis_dir):

    alist = [a for a in os.listdir(json_file) if '.json' in a]

    for i in range(0, len(alist)):
        path = os.path.join(json_file, alist[i])
        data = json.load(open(path))

        out_dir = osp.basename(path).replace('.', '_')
        out_dir = osp.join(labelme_vis_dir, out_dir)
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')

        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in data['shapes']:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))

        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
        captions = ['{}: {}'.format(lv, ln)
                    for ln, lv in label_name_to_value.items()]
        lbl_viz = utils.draw_label(lbl, img, captions)

        Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
        # utils.lblsave(osp.join('G:/cashgt/', path[0:len(path) - 4] + 'png'), lbl)
        Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir)


if __name__ == '__main__':

    json_file = r'C:\Users\15974\Desktop\1'
    labelme_vis_dir = r'C:\Users\15974\Desktop\2'
    if not os.path.exists(labelme_vis_dir):
        os.makedirs(labelme_vis_dir)
    main(json_file, labelme_vis_dir)