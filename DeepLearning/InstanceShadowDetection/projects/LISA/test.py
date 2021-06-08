# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import math
import cv2
import tqdm
import skimage.io as io
import numpy as np
import skimage.transform as tf
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo


from LISA import add_lisa_config 

def distance(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return int(math.sqrt((a * a) + (b * b)))

kernels = np.array([[0,1],[1,0],[-1,0],[0,-1]])



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            # args.input = glob.glob(os.path.expanduser(args.input[0]))   
            args.input = glob.glob(os.path.join(args.input[0], "*.jpg"))
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions[0][0]["instances"]), time.time() - start_time
                )
            )
            ins,rel = predictions
            information = ins[0]['instances']
            boxes = information.pred_boxes.tensor.numpy()
            classes = information.pred_classes.cpu().numpy()
            associations = information.pred_associations
            number_of_pair = max(associations)
            src = cv2.imread(path, cv2.IMREAD_COLOR)
    
            if not os.path.isdir("./myTest"):
                os.mkdir("./myTest")

            for i, box in enumerate(boxes):
                if associations[i] is not 0:
                    src = cv2.rectangle(src, tuple(map(int,box[:2])), tuple(map(int, box[2:])), tuple(map(int, np.random.choice(range(256), size=3))), 3)
            cv2.imwrite(os.path.join("./myTest", os.path.split(path)[-1]), src)
            

            # #################################################################
            # # Using delta Light direction Draw
            # src = cv2.imread(path, cv2.IMREAD_COLOR)
            # if not os.path.isdir("./delta_result"):
            #     os.mkdir("./delta_result")

            # for i, box in enumerate(boxes):
            #     if associations[i] is not 0:
            #         src = cv2.rectangle(src, tuple(map(int,box[:2])), tuple(map(int, box[2:])), tuple(map(int, np.random.choice(range(256), size=3))), 3)

            # middle_points = []
            # slopes = []
            # deltas = []
            # light_points = []

            # for pair in range(1,1+number_of_pair):
            #     points = [0,0]
            #     delta = [0,0]
            #     for a_index, association in enumerate(associations):
            #         if association == pair:
            #             x = int((boxes[a_index][0] + boxes[a_index][2])/2)
            #             y = int((boxes[a_index][1] + boxes[a_index][3])/2)
            #             dx = int(abs(boxes[a_index][0] - x))
            #             dy = int(abs(boxes[a_index][1] - y))
            #             points[classes[a_index]] = (x, y)
            #             delta[classes[a_index]] = (dx, dy)
            #     middle_points.append(points)
            #     deltas.append(delta)


            # for middle_point, delta in zip(middle_points, deltas):
            #     object_middle = middle_point[0]
            #     shadow_middle = middle_point[1]
            #     object_delta = delta[0]
            #     shadow_delta = delta[1]
            #     distances_o2s = []
            #     distances_s2o = []

            #     for kernel in kernels:
            #         object_move_point = object_middle + kernel * object_delta
            #         shadow_move_point = shadow_middle + kernel * shadow_delta

            #         d_o2s = distance(object_move_point, shadow_middle)
            #         d_s2o = distance(shadow_move_point, object_middle)
            #         distances_o2s.append(d_o2s)
            #         distances_s2o.append(d_s2o)

            #     max_index_object = distances_o2s.index(max(distances_o2s))
            #     max_index_shadow = distances_s2o.index(max(distances_s2o))

            #     object_target = object_middle + kernels[max_index_object] * object_delta
            #     shadow_target = shadow_middle + kernels[max_index_shadow] * shadow_delta
            #     light_points.append([object_target, shadow_target])

            # for light in light_points:
            #     cv2.arrowedLine(src, light[1], light[0], tuple(map(int, np.random.choice(range(256), size=3))), 3)
            # cv2.imwrite(os.path.join("./delta_result", os.path.split(path)[-1]), src)