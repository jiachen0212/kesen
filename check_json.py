import json
import os
from re import L


def json_label_check(js_path, defects, defcet_nums):

    try:
        data = json.load(open(js_path, 'r'))
    except:
        return 

    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            # def_ = cls_['labels'][0]
            def_ = cls_['label']
            defcet_nums[defects.index(def_)] += 1


def defect2ignore():
    js_dir= r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\DL\kesen\data\隧道\cuted_dir\黑线\银白色\0523'
    js_paths = [os.path.join(js_dir, a) for a in os.listdir(js_dir) if '.json' in a]
    for js_path in js_paths:
        try:
            data = json.load(open(js_path, 'r'))
        except:
            continue 
        data1 = data.copy()
        data1['shapes'] = []
        if len(data['shapes']) > 0:
            for cls_ in data['shapes']:
                cls_1 = cls_.copy()
                def_ = cls_['label']
                if def_ == "huichen":
                    
                    cls_1['label'] = "ignore"
                    assert len(cls_1['labels']) == 1
                    cls_1['labels'] = ["ignore"]
                    data1['shapes'].append(cls_1)
                else:
                    data1['shapes'].append(cls_)
        # a, b = 0, 0
        # for cls_ in data1['shapes']:
        #     def_ = cls_['label']
        #     if def_ == "ignore":
        #         a += 1
        #     elif def_ == "huichen":
        #         b += 1
        # print(a, b)
        # js_path1 = os.path.join(r'C:\Users\15974\Desktop\111', os.path.basename(js_path))
        data = json.dumps(data1, indent=4)
        with open(js_path, 'w') as js_file:
            js_file.write(data)

def calcuate_defect_nums():

    defects = ["heixian", "fushidian", "huichen", "zangwu", "ignore", "znagwu"]
    defcet_nums = [0]*len(defects)

    js_dir = r'C:\Users\15974\Desktop\20220526_JSON(2)\20220526_JSON'
    js_paths = [os.path.join(js_dir, a) for a in os.listdir(js_dir)]
    for js_path in js_paths:
        json_label_check(js_path, defects, defcet_nums)

    a = ''
    for ind, b in enumerate(defects):
        a += '{}: {}, '.format(b, defcet_nums[ind])
    print(a)


if __name__ == '__main__':
    # calcuate_defect_nums()
    defect2ignore()
