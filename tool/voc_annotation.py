# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import json
from collections import defaultdict
from tqdm import tqdm
import os

"""hyper parameters"""
ROOT_DIR = '/home/deeplab307/Documents/Anaconda/Shiang/IL/'
json_file_path = os.path.join(ROOT_DIR, 'dataset/voc2007/annotations/voc2007_trainval.json')
images_dir_path = os.path.join(ROOT_DIR, 'dataset/voc2007/images')
output_path = os.path.join(ROOT_DIR, 'dataset/voc2007/annotations/voc2007_yolov4_trainval.txt')

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

"""generate labels"""
images = data['images']
annotations = data['annotations']
for ant in tqdm(annotations):
    id = ant['image_id']
    name = os.path.join(images_dir_path, '{:06d}.jpg'.format(id))
    cat = ant['category_id'] - 1

    name_box_id[name].append([ant['bbox'], cat])

"""write to txt"""
with open(output_path, 'w') as f:
    for key in tqdm(name_box_id.keys()):
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
