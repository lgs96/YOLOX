#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import json

import torch
from collections import ChainMap, defaultdict

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from pycocotools.coco import COCO
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
import datetime
from pycocotools.cocoeval import COCOeval
from coco_categories import get_categories
import copy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='bullet')
    parser.add_argument("-n", "--name", type=str, default='yolox-l', help="model name")

    parser.add_argument(
        "--path", default="/home/sonic/Desktop/zts/jpg_result/downtown_miami.mp4_720p/", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    
    parser.add_argument(
        "--save_folder_name", type=str, default='downtown_miami.mp4_720p', help="save folder name")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default='./pretrained_model/yolox_l.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument('--tsize', default=None, type=int, help="test img size")

    
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

# Predictor also measure the performance
class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        ## goodsol: save gpu time
        self.save_folder_name = ''    
        self.image_index = 0
        self.ann_id = 0
        self.network_detect_queue = []
        self.network_counter_queue = []

        self.evaluated_frame_num = 0
        self.sum_mAP_50_95 = 0
        self.sum_mAP_50 = 0

        self.mAP_file_name = "./perf/mAP_stats.txt"

        ## accumulated results for ground truth and detection result
        self.acc_gt_result = []
        self.acc_dt_result = []
    
        self.null_detect_data = {
                "image_id": int(0),
                "category_id": None,
                "bbox": [],
                "score": None,
                "segmentation": [],
                "id": int(0),
            }

        self.detect_result = [self.null_detect_data]

        self.coco = COCO(os.path.join("/home/sonic/Desktop/VideoAnalytics/YOLOX/coco/annotations/instances_val2017.json"))
        self.class_ids = sorted(self.coco.getCatIds())

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            self.record_infer_time(round(time.time() - t0, 4))
            covert_outputs = copy.deepcopy(outputs)
            covert_img_info = copy.deepcopy(img_info)
            self.record_detect_json(covert_outputs, covert_img_info)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

    def record_infer_time (self, time):
        if self.image_index == 0:
            f = open(self.save_folder_name+'/gpu_latency.txt', 'w')
        else:
            f = open(self.save_folder_name+'/gpu_latency.txt', 'a')
        f.write(str(self.image_index) + '\t' + str(time) +'\n')

    def convert_to_coco_format(self, output, img_info, img_id):
        gt_dict = {}
        dt_dict = {}

        gt_image_list = []
        dt_image_list = []

        gt_ann_list = []
        dt_ann_list = []

        image_wise_data = defaultdict(dict)

        img_h = img_info["height"]
        img_w = img_info["width"]

        if output is None:
            return
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            self.test_size[0] / float(img_h), self.test_size[1] / float(img_w)
        )
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        image_wise_data.update({
            int(img_id): {
                "bboxes": [box.numpy().tolist() for box in bboxes],
                "scores": [score.numpy().item() for score in scores],
                "categories": [
                    self.class_ids[int(cls[ind])]
                    for ind in range(bboxes.shape[0])
                ],
            }
        })

        bboxes = xyxy2xywh(bboxes)

        file_name = "frame_" + str(img_id)
        date_time = datetime.datetime.now()
        formatted_datetime = date_time.isoformat()

        # "images"
        pred_image = {
            "id": int(img_id),
            "width": int(img_w),
            "height": int(img_h),
            "file_name": str(file_name),
            "license": int(0),
            "flickr_url": None,
            "coco_url": None,
            "data_captured": formatted_datetime,
        }

        dt_image_list.append(pred_image)
        gt_image_list.append(pred_image)

        dt_dict['images'] = dt_image_list
        gt_dict['images'] = gt_image_list

        # "annotations"
        for ind in range(bboxes.shape[0]):
            label = self.class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
                "iscrowd": int(0),
                "area" : float(0),
                "id": int(self.ann_id),
            }  # COCO json format
            
            '''
            detect_data = {
                "image_id": int(img_id + 4),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
                "id": int(self.ann_id),
            }  # COCO json format
            '''

            self.ann_id += 1
            gt_ann_list.append(pred_data)

        if bboxes.shape[0] > 0:
            self.set_detect_result(gt_ann_list)
        else:
            self.null_detect_data['image_id'] = int(img_id)
            self.set_detect_result([self.null_detect_data])
        dt_ann_list = self.detect_result 

        gt_dict["annotations"] = gt_ann_list
        gt_dict["categories"] = get_categories()

        if img_id == 0:
            self.acc_gt_result = gt_dict 
            self.acc_dt_result = gt_ann_list
        else:
            self.acc_gt_result["annotations"].extend(gt_ann_list)
            self.acc_dt_result.extend(dt_ann_list)

        return gt_dict, dt_ann_list

    def record_detect_json (self, outputs, img_info, is_real_time = True):
        output = outputs[0]
        bbox_gt, bbox_dt = self.convert_to_coco_format (output, img_info, self.image_index)

        img_id = bbox_gt['annotations'][0]['image_id']
        for bbox in bbox_dt:  
            bbox['image_id'] = img_id

        if is_real_time:
            self.evaluated_frame_num += 1
            file_name = 'result.json'
        else:
            file_name = 'acc_result.json'

        json.dump(bbox_gt, open(self.save_folder_name+'/gt_' + file_name, 'w'))
        json.dump(bbox_dt, open(self.save_folder_name+'/dt_' + file_name, 'w'))
        
        self.evaluation(self.save_folder_name+'/gt_' + file_name, self.save_folder_name+'/dt_' + file_name)

    def set_result_delay (self):
        return


    ## 230202 TODO: network_detect_queue append all bbox on index 0 elemenet.. 
    def set_detect_result (self, gt_ann_list):
        ## intention: read e2e latency file
        # to transmit only detect result that device sent
        #if frame_id == gt_ann_list[0][image_id]:

        my_list = copy.deepcopy(gt_ann_list)

        delay = 1 #frames, TODO: set_result_delay function

        self.network_counter_queue.append(delay + 1)
        self.network_detect_queue.append(my_list)

        print('Counter queue: ', self.network_counter_queue)

        for i, counter in enumerate(self.network_counter_queue):
            self.network_counter_queue[i] = counter - 1
            print('Net queue len: ', len(self.network_detect_queue[i]))
        if len(self.network_counter_queue) > 0 and self.network_counter_queue[0] == 0:
            self.network_counter_queue.pop(0)
            self.detect_result = self.network_detect_queue.pop(0)
        

    def evaluation (self, gt_json, dt_json):
        cocoGt = COCO(gt_json)
        cocoDt = cocoGt.loadRes(dt_json)

        coco_eval = COCOeval(cocoGt, cocoDt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        try:
            if float(coco_eval.stats[0]) != 0:
                self.sum_mAP_50_95 += coco_eval.stats[0]
                self.sum_mAP_50 += coco_eval.stats[1]
                stats = str(coco_eval.stats[0]) + '\t' + str(coco_eval.stats[1]) + '\t' + str(self.sum_mAP_50_95/self.evaluated_frame_num) + '\t' + str(self.sum_mAP_50/self.evaluated_frame_num)
                print(stats)
                if self.evaluated_frame_num == 1:
                    with open(self.mAP_file_name, "w") as f:
                        f.write("Current mAP_50_95" + '\t' + "Current mAP_50" + '\t' + "Average mAP_50_95" + '\t' + "Average mAP_50" +'\n'
                        + stats + '\n')
                else:
                    with open(self.mAP_file_name, "a") as f:
                        f.write(stats + '\n')
            else:
                logger.info("No box detected..")
        except:
            logger.info("No box detected..")

# defined by Goodsol.
def single_image(predictor, vis_folder, image_name, current_time, save_result, image_index):
    predictor.image_index = image_index
    predictor.save_folder_name = vis_folder
    outputs, img_info = predictor.inference(image_name)
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    if save_result:
        save_folder = os.path.join(vis_folder)
        os.makedirs(save_folder, exist_ok=True)
        save_file_name = os.path.join(save_folder, os.path.basename(image_name))
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_image)
    ch = cv2.waitKey(0)


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, args.save_folder_name)
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    frame_index = 0
    while(True):
        current_path = ''

        current_path = args.path +'frame_'+str(frame_index)+'.jpg'
        logger.info('Current path: {}'.format(str(current_path)))
        current_time = time.localtime()
        single_image(predictor, vis_folder, current_path, current_time, args.save_result, frame_index)
        frame_index += 1
        #except:
        #    logger.info('No remaining frame for inference')
        #    break
        # we only evaluate on sequential image dataset
        #if args.demo == "image":
        #    image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
        #elif args.demo == "video" or args.demo == "webcam":
        #    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
