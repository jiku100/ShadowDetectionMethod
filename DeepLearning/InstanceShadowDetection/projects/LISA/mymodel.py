# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import skimage.io as io
import numpy as np
import skimage.transform as tf
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from PIL import Image

from resnext.resnext101_regular import ResNeXt101

from LISA import add_lisa_config 

to_pil = transforms.ToPILImage()

class _AttentionModule(nn.Module):

    def __init__(self):
        super(_AttentionModule, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=2, padding=2, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=3, padding=3, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=4, padding=4, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True).clone()
        block2 = F.relu(self.block2(block1) + block1, True).clone()
        block3 = F.sigmoid(self.block3(block2) + self.down(block2))
        return block3
class BDRAR(nn.Module):
    def __init__(self):
        super(BDRAR, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.refine3_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine2_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine1_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention3_hl = _AttentionModule()
        self.attention2_hl = _AttentionModule()
        self.attention1_hl = _AttentionModule()

        self.refine2_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine4_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine3_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention2_lh = _AttentionModule()
        self.attention3_lh = _AttentionModule()
        self.attention4_lh = _AttentionModule()

        self.fuse_attention = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 1)
        )

        self.predict = nn.Sequential(
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
        refine3_hl_0 = F.relu(self.refine3_hl(torch.cat((down4, down3), 1)) + down4, True).clone()
        refine3_hl_0 = (1 + self.attention3_hl(torch.cat((down4, down3), 1))) * refine3_hl_0
        refine3_hl_1 = F.relu(self.refine3_hl(torch.cat((refine3_hl_0, down3), 1)) + refine3_hl_0, True).clone()
        refine3_hl_1 = (1 + self.attention3_hl(torch.cat((refine3_hl_0, down3), 1))) * refine3_hl_1

        refine3_hl_1 = F.upsample(refine3_hl_1, size=down2.size()[2:], mode='bilinear')
        refine2_hl_0 = F.relu(self.refine2_hl(torch.cat((refine3_hl_1, down2), 1)) + refine3_hl_1, True).clone()
        refine2_hl_0 = (1 + self.attention2_hl(torch.cat((refine3_hl_1, down2), 1))) * refine2_hl_0
        refine2_hl_1 = F.relu(self.refine2_hl(torch.cat((refine2_hl_0, down2), 1)) + refine2_hl_0, True).clone()
        refine2_hl_1 = (1 + self.attention2_hl(torch.cat((refine2_hl_0, down2), 1))) * refine2_hl_1

        refine2_hl_1 = F.upsample(refine2_hl_1, size=down1.size()[2:], mode='bilinear')
        refine1_hl_0 = F.relu(self.refine1_hl(torch.cat((refine2_hl_1, down1), 1)) + refine2_hl_1, True).clone()
        refine1_hl_0 = (1 + self.attention1_hl(torch.cat((refine2_hl_1, down1), 1))) * refine1_hl_0
        refine1_hl_1 = F.relu(self.refine1_hl(torch.cat((refine1_hl_0, down1), 1)) + refine1_hl_0, True).clone()
        refine1_hl_1 = (1 + self.attention1_hl(torch.cat((refine1_hl_0, down1), 1))) * refine1_hl_1

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        refine2_lh_0 = F.relu(self.refine2_lh(torch.cat((down1, down2), 1)) + down1, True).clone()
        refine2_lh_0 = (1 + self.attention2_lh(torch.cat((down1, down2), 1))) * refine2_lh_0
        refine2_lh_1 = F.relu(self.refine2_lh(torch.cat((refine2_lh_0, down2), 1)) + refine2_lh_0, True).clone()
        refine2_lh_1 = (1 + self.attention2_lh(torch.cat((refine2_lh_0, down2), 1))) * refine2_lh_1

        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        refine3_lh_0 = F.relu(self.refine3_lh(torch.cat((refine2_lh_1, down3), 1)) + refine2_lh_1, True).clone()
        refine3_lh_0 = (1 + self.attention3_lh(torch.cat((refine2_lh_1, down3), 1))) * refine3_lh_0
        refine3_lh_1 = F.relu(self.refine3_lh(torch.cat((refine3_lh_0, down3), 1)) + refine3_lh_0, True).clone()
        refine3_lh_1 = (1 + self.attention3_lh(torch.cat((refine3_lh_0, down3), 1))) * refine3_lh_1

        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')
        refine4_lh_0 = F.relu(self.refine4_lh(torch.cat((refine3_lh_1, down4), 1)) + refine3_lh_1, True).clone()
        refine4_lh_0 = (1 + self.attention4_lh(torch.cat((refine3_lh_1, down4), 1))) * refine4_lh_0
        refine4_lh_1 = F.relu(self.refine4_lh(torch.cat((refine4_lh_0, down4), 1)) + refine4_lh_0, True).clone()
        refine4_lh_1 = (1 + self.attention4_lh(torch.cat((refine4_lh_0, down4), 1))) * refine4_lh_1

        refine3_hl_1 = F.upsample(refine3_hl_1, size=down1.size()[2:], mode='bilinear')
        predict4_hl = self.predict(down4)
        predict3_hl = self.predict(refine3_hl_1)
        predict2_hl = self.predict(refine2_hl_1)
        predict1_hl = self.predict(refine1_hl_1)

        predict1_lh = self.predict(down1)
        predict2_lh = self.predict(refine2_lh_1)
        predict3_lh = self.predict(refine3_lh_1)
        predict4_lh = self.predict(refine4_lh_1)

        fuse_attention = F.sigmoid(self.fuse_attention(torch.cat((refine1_hl_1, refine4_lh_1), 1)))
        fuse_predict = torch.sum(fuse_attention * torch.cat((predict1_hl, predict4_lh), 1), 1, True)

        predict4_hl = F.upsample(predict4_hl, size=x.size()[2:], mode='bilinear')
        predict3_hl = F.upsample(predict3_hl, size=x.size()[2:], mode='bilinear')
        predict2_hl = F.upsample(predict2_hl, size=x.size()[2:], mode='bilinear')
        predict1_hl = F.upsample(predict1_hl, size=x.size()[2:], mode='bilinear')
        predict1_lh = F.upsample(predict1_lh, size=x.size()[2:], mode='bilinear')
        predict2_lh = F.upsample(predict2_lh, size=x.size()[2:], mode='bilinear')
        predict3_lh = F.upsample(predict3_lh, size=x.size()[2:], mode='bilinear')
        predict4_lh = F.upsample(predict4_lh, size=x.size()[2:], mode='bilinear')
        fuse_predict = F.upsample(fuse_predict, size=x.size()[2:], mode='bilinear')

        if self.training:
            return fuse_predict, predict1_hl, predict2_hl, predict3_hl, predict4_hl, predict1_lh, predict2_lh, predict3_lh, predict4_lh
        return F.sigmoid(fuse_predict)

class Mymodel(nn.Module):
    def __init__(self, cfg):
        super(Mymodel, self).__init__() 
        self.LISA = DefaultPredictor(cfg)
        self.BDRAR = BDRAR()
        self.BDRAR.eval()

    def forward(self, img):
        predictions = self.LISA(img)
        ins,rel = predictions
        information = ins[0]['instances']
        boxes = information.pred_boxes.tensor.numpy()
        classes = information.pred_classes.cpu().numpy()
        associations = information.pred_associations
        number_of_pair = max(associations)
        for i, box in enumerate(boxes):
            if classes[i] == 1:
                w = int(box[2] - box[0])
                h = int(box[3] - box[1])
                roi = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
                roi = cv2.resize(roi, dsize=(256, 256), interpolation=cv2.INTER_AREA)
                roi = torch.as_tensor(roi.astype("float32").transpose(2, 0, 1)).unsqueeze(0)
                roi_shadow = self.BDRAR(roi)
                roi_shadow = np.array(transforms.Resize((h,w))(to_pil(roi_shadow.data.squeeze(0).cpu())))
                Image.fromarray(roi_shadow).save("./myTest/a.jpg")

        return predictions 

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

    model = Mymodel(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))   
            #args.input = glob.glob(os.path.join(args.input[0], "*.jpg"))
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions = model(img)
            
            # logger.info(
            #     "{}: detected {} instances in {:.2f}s".format(
            #         path, len(predictions[0][0]["instances"]), time.time() - start_time
            #     )
            # )
            # ins,rel = predictions
            # information = ins[0]['instances']
            # boxes = information.pred_boxes.tensor.numpy()
            # classes = information.pred_classes.cpu().numpy()
            # associations = information.pred_associations
            # number_of_pair = max(associations)
           