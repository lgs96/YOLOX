#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import ChainMap, defaultdict

## 0. get outputs, convert it into coco format, and generate cocoGt


## 1. For every output log, generate corresponding cocoDt


## 2. generate cocoGt, cocoDt for real-time mode and accumulate mode


## 3. Evaluate with COCOEval

frame_interval = 33

class PerfMeasurement:

    def __init__(self, e2e_latency_file = None, gpu_latency_file = None, img_size = (640,640), annType):
        self.e2e_f = 0
        self.gpu_f = 0
        self.img_size = img_size
        self.annType = annType # one of 'segm', 'bbox', 'keypoints'
        self.bbox_set = []
        if e2e_latency_file not None:
            self.e2e_f = open(e2e_latency_file, 'r')
        if gpu_latency_file not None:
            self.gpu_f = open(gpu_latency_file, 'r')

    def convert_to_coco_format(self, output, info_img, img_id):
        data_list = []
        image_wise_data = defaultdict(dict)

        img_h = info_img[0]
        img_w = info_img[1]
        
        if output is None:
            continue
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
        )
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        image_wise_data.update({
            int(img_id): {
                "bboxes": [box.numpy().tolist() for box in bboxes],
                "scores": [score.numpy().item() for score in scores],
                "categories": [
                    self.dataloader.dataset.class_ids[int(cls[ind])]
                    for ind in range(bboxes.shape[0])
                ],
            }
        })

        bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            label = self.dataloader.dataset.class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)

        return data_list

    # generate json file for instant evaluation and recorded evaluation
    def generate_cocoGt (self, bbox_list):
        self.bbox_set.append(bbox_list)
        json.dump(bbox_json, open("./perf/current_cocoGt.json", "w"))
        return COCO("./perf/current_cocoGt.json")

    def set_current_detect_result (self):

    # generate json file for instant evaluation and recorded evaluation
    def generate_cocoDt (self):
        json.dump(bbox_list, open("./perf/current_cocoDt.json", "w"))
        return self.cocoGt.loadRes("./perf/current_cocoDt.json")

    # evaluate single image    
    def evaluate_single (self, output, info_img, id):
        bbox_list = self.convert_to_coco_format(output, info_img, id):

        self.cocoGt = self.generate_cocoGt (bbox_list)
        self.cocoDt = self.generate_cocoDt ()

        cocoEval = COCOeval(self.cocoGt, self.cocoDt, self.annType)
        cocoEval.evaluate()
        return cocoEval.stats[0], cocoEval.stats[1]

    def read_e2e_list (self):
        self.e2e_f.readline()

    ## need additioanl codes
    def evaluate_batch (self,)
        cocoGt = 
        cocoDt =
    
        cocoEval = COCOeval(self.cocoGt, self.cocoDt, self.annType)
        cocoEval.evaluate()
        cocoEval.accumulate()

