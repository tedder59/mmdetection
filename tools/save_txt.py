#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author        :   yuyong
# @Created Time  :   2019年01月28日 星期一 14时43分01秒
# @File Name     :   show.py
import argparse
import os
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result
import cv2
import numpy as np
from mmdet.core import get_classes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/mapbar1/yuyong/data/qinghua/images')
    parser.add_argument('--image_save', type=str, default='/mapbar1/yuyong/data/qinghua/result')
    parser.add_argument('--result_txt', type=str, default='/mapbar1/yuyong/data/qinghua/results.txt')

    args = parser.parse_args()

    return args


def det_bboxes(img,
               bboxes,
               labels,
               score_thr=0,
               bbox_color='green',
               text_color='green',
               thickness=2,
               font_scale=0.5,
               show=True,
               win_name='',
               wait_time=0,
               out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = mmcv.color_val(bbox_color)
    text_color = mmcv.color_val(text_color)

    ret_str = ''

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = str(label)
        dataset_name = 'sign'
        class_names = get_classes(dataset_name)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        w = bbox_int[2] - bbox_int[0]
        h = bbox_int[3] - bbox_int[1]
        ss = ' '+str(left_top[0])+','+str(left_top[1])+','+str(w)+','+str(h)
        if len(bbox) > 4:
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color)
        ret_str += ss


    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return ret_str



def get_result(img, result, score_thr=0.3, out_file=None):
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    ret_str = det_bboxes(img.copy(), 
                         bboxes, 
                         labels,
                         score_thr=0.3,
                         show=False,
                         out_file=out_file)

    return ret_str


def main():
    args = parse_args()

    config_file = '/home/yuyong/git/mmdetection/version/sign/cascade_mask_rcnn_x101_32x4d_fpn_1x.py'
    checkpoint_file = '/home/yuyong/git/mmdetection/version/sign/epoch_20.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file)
    #images_ids = [d.name for d in os.scandir(args.image_dir) if d.name.endswith('.jpg')]
    
    val_txt = '/mapbar1/yuyong/data/qinghua/train.txt'
    with open(val_txt, 'r') as f:
        images_ids = f.readlines()
    images_ids = [x.strip('\n') + '.jpg' for x in images_ids]

    with open(args.result_txt, 'w') as f:
        for image_id in images_ids:
            img_path = os.path.join(args.image_dir, image_id)
            img = mmcv.imread(img_path)
            #img = cv2.resize(img, (1280, 720))
            if img is None:
                continue
            result = inference_detector(model, img)
            out_file = os.path.join(args.image_save, image_id)
            ret_str = get_result(img, result, out_file=out_file)
            ret_str = image_id + ret_str
            print(ret_str)
            f.write(ret_str)
            f.write("\n")
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
