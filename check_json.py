import json
import os
import random


def json_label_check(js_path, defects, defcet_nums):

    try:
        data = json.load(open(js_path, 'r'))
    except:
        print('bad json', js_path)
        return 

    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            # def_ = cls_['labels'][0]
            def_ = cls_['label']
            defcet_nums[defects.index(def_)] += 1


def defect2ignore(cur_defcet):
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
                if def_ == cur_defcet:
                    
                    cls_1['label'] = "ignore"
                    assert len(cls_1['labels']) == 1
                    cls_1['labels'] = ["ignore"]
                    data1['shapes'].append(cls_1)
                else:
                    data1['shapes'].append(cls_)
        data = json.dumps(data1, indent=4)
        with open(js_path, 'w') as js_file:
            js_file.write(data)


def defect_random2_ignore(cur_defect, js_dir):
    js_paths = [os.path.join(js_dir, a) for a in os.listdir(js_dir) if '.json' in a]
    for js_path in js_paths:
        try:
            data = json.load(open(js_path, 'r'))
        except:
            print('bad json')
            continue 
        data1 = data.copy()
        data1['shapes'] = []
        if len(data['shapes']) > 0:
            for cls_ in data['shapes']:
                cls_1 = cls_.copy()
                def_ = cls_['label']
                if def_ == cur_defect:
                    seed = random.random()
                    if seed > 0.2:
                        cls_1['label'] = "ignore"
                        assert len(cls_1['labels']) == 1
                        cls_1['labels'] = ["ignore"]
                        data1['shapes'].append(cls_1)
                    else:
                        data1['shapes'].append(cls_)
                else:
                    # 除cur_defect之外的其他缺陷
                    data1['shapes'].append(cls_)

        js_path1 = os.path.join(r'C:\Users\15974\Desktop\1', os.path.basename(js_path))
        data1_ = json.dumps(data1, indent=4)
        with open(js_path1, 'w') as js_file:
            js_file.write(data1_)

def calcuate_defect_nums(js_dir, defects):

    defcet_nums = [0]*len(defects)

    js_paths = [os.path.join(js_dir, a) for a in os.listdir(js_dir) if '.json' in a]
    for js_path in js_paths:
        json_label_check(js_path, defects, defcet_nums)

    a = ''
    for ind, b in enumerate(defects):
        a += '{}: {}, '.format(b, defcet_nums[ind])
    print(a)


if __name__ == '__main__':
    
    dir_ = r'C:\Users\15974\Desktop\【19】滴酸异色大面-工位4-同轴光_已完成\【19】滴酸异色大面-工位4-同轴光_已完成'
    # defects = ["heixian", "fushidian", "huichen", "zangwu", "ignore", "znagwu"]
    defects = ["disuanyise-dm"]
    calcuate_defect_nums(dir_, defects)
    # defect_random2_ignore('fushidian', dir_)
    # dir_ =  r'C:\Users\15974\Desktop\1'   
    # calcuate_defect_nums(dir_)
   

                

                    
    
    


