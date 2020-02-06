#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author        :   yuyong
# @Created Time  :   2019年01月28日 星期一 14时43分01秒
# @File Name     :   show.py
import argparse
import os
import mmcv
import cv2
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result

import time
from pprint import pprint

SIGN_CLASSES = ('traffic3', 'traffic4', 'traffic3-back', 'traffic4-back', 'circle', 'circle-back')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/home/yuyong/data/images/')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    config_file = '/home/yuyong/git/mmdetection/version/sign/cascade_mask_rcnn_x101_32x4d_fpn_1x.py'
    checkpoint_file = '/home/yuyong/git/mmdetection/version/sign/epoch_20.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file)

    #print(model)

    # test a list of images
    images_ids = [d.name for d in os.scandir(args.image_dir)]
    #with open(val_txt, 'r') as f:
    #    files = f.readlines()
    #images_ids = [x.strip('\n')+'.jpg' for x in files]
    for image_id in images_ids:
        img_path = os.path.join(args.image_dir, image_id)
        img_path = "/home/yuyong/git/mmdetection/shangdi_b_0.jpg"
        img = mmcv.imread(img_path)
        img = cv2.resize(img, (1280, 720))
        result = inference_detector(model, img)
        model.show_result(img, result)
        #show_result(img, result, SIGN_CLASSES)
        #show_result(img, result)
        exit()



if __name__ == '__main__':
    main()
