#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author        :   yuyong
# @Created Time  :   2019年01月22日 星期二 16时53分05秒
# @File Name     :   sign.py
import os.path as osp
import json

import mmcv
import numpy as np
from pycocotools.mask import _mask as maskUtils 

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class HadSignDataset(CustomDataset):

    CLASSES = ('traffic3', 'traffic4', 'circle')
    traffic3_CLASSES = ('traffic3', 'traffic3-occ-partially', 'traffic3rev', 'traffic3rev-occ-partially')
    traffic4_CLASSES = ('traffic4', 'traffic4-occ-partially', 'traffic4y', 'traffic4y-occ-partially')
    Dets_CLASSES = CLASSES + traffic3_CLASSES + traffic4_CLASSES

    NAME = 'sign'

    def __init__(self, **kwargs):
        super(HadSignDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'images/{}.jpg'.format(img_id)
            json_path = osp.join(self.img_prefix, 'jsons',
                                 '{}.json'.format(img_id))
            with open(json_path, 'r') as f:
                data = json.load(f)
                width = int(data['imageWidth'])
                height = int(data['imageHeight'])
            img_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        json_path = osp.join(self.img_prefix, 'jsons',
                             '{}.json'.format(img_id))
        return self._parse_ann_info(json_path)


    def _parse_ann_info(self, json_path):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        # mask
        gt_masks = []
        gt_mask_polys = []
        gt_poly_lens = []
        with open(json_path, 'r') as f:
            data = json.load(f)
            shapes = data['shapes']
            h = int(data['imageHeight'])
            w = int(data['imageWidth']) 
            for shape in shapes:
                sub = [v for points in shape['points'] for v in points]
                if shape['shape_type'] == 'polygon':
                    bbox = [
                        int(min(sub[::2])),
                        int(min(sub[1::2])),
                        int(max(sub[::2])),
                        int(max(sub[1::2]))
                    ]
                    mask_polys = [sub]
                else:
                    """
                    sub[0] = float(sub[0])
                    sub[1] = float(sub[1])
                    sub[2] = float(sub[2])
                    sub[3] = float(sub[3])
                    """
                    bbox = [
                        int(sub[0]),
                        int(sub[1]),
                        int(sub[2]),
                        int(sub[3])
                    ]
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    mask_polys = [[bbox[0], bbox[1], bbox[0]+bbox_w, bbox[1],
                                  bbox[2], bbox[3], bbox[0], bbox[3]+bbox_h]]
                #if shape['label'] in self.cat2label.keys():
                if shape['label'] in self.Dets_CLASSES:
                    if shape['label'] in self.traffic3_CLASSES:
                        shape['label'] = self.traffic3_CLASSES[0]
                    if shape['label'] in self.traffic4_CLASSES:
                        shape['label'] = self.traffic4_CLASSES[0]
                    label = self.cat2label[shape['label']]
                    bboxes.append(bbox)
                    labels.append(label)
                    # with mask
                    rles = maskUtils.frPyObjects(mask_polys, h, w)
                    rle = maskUtils.merge(rles)
                    gt_masks.append(maskUtils.decode([rle])[:,:,0])
                    poly_lens = [len(p) for p in mask_polys]
                    gt_mask_polys.append(mask_polys)
                    gt_poly_lens.extend(poly_lens)
                    
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        # with mask
        ann['masks'] = gt_masks
        # poly format is not used in the current implementation
        ann['mask_polys'] = gt_mask_polys
        ann['poly_lens'] = gt_poly_lens
        
        return ann








