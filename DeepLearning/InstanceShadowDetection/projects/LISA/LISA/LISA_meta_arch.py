#  Copyright (c) Tianyu Wang. All Rights Reserved.
import logging
import torch
from torch import nn
import cv2
import numpy as np
import torch.nn.functional as F
from detectron2.structures import ImageList
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.utils.logger import log_first_n
from detectron2.layers import paste_masks_in_image
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess, matchor, combine_association
from .LISA_rpn  import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import __all__, GeneralizedRCNN
from detectron2.utils.registry import Registry
__all__.append("LISARCNN")

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
        out1 = nn.ReLU()(self.block1(x) + x)
        out2 = nn.ReLU()(self.block2(out1) + out1)
        out3 = nn.Sigmoid()(self.block3(out2) + self.down(out2))
        return out3

class BDRAR_BACKBONE(nn.Module):
    def __init__(self):
        super(BDRAR_BACKBONE, self).__init__()

        self.down5 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.refine4_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
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

        self.attention4_hl = _AttentionModule()
        self.attention3_hl = _AttentionModule()
        self.attention2_hl = _AttentionModule()
        self.attention1_hl = _AttentionModule()

        self.refine1_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
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
        

        self.attention1_lh = _AttentionModule()
        self.attention2_lh = _AttentionModule()
        self.attention3_lh = _AttentionModule()
        self.attention4_lh = _AttentionModule()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1)
        )


        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, backbone):
        fpn1 = backbone[0]
        fpn2 = backbone[1]
        fpn3 = backbone[2]
        fpn4 = backbone[3]
        fpn5 = backbone[4]

        down1 = self.down1(fpn1)
        down2 = self.down2(fpn2)
        down3 = self.down3(fpn3)
        down4 = self.down4(fpn4)
        down5 = self.down5(fpn5)

        down5 = F.upsample(down5, size=down4.size()[2:], mode='bilinear')
        refine4_hl_0 = F.relu(self.refine4_hl(torch.cat((down5, down4), 1)) + down5, True)
        refine4_hl_0 = (1 + self.attention4_hl(torch.cat((down5, down4), 1))) * refine4_hl_0
        refine4_hl_1 = F.relu(self.refine4_hl(torch.cat((refine4_hl_0, down4), 1)) + refine4_hl_0, True)
        refine4_hl_1_feature = (1 + self.attention4_hl(torch.cat((refine4_hl_0, down4), 1))) * refine4_hl_1

        refine4_hl_1 = F.upsample(refine4_hl_1_feature, size=down3.size()[2:], mode='bilinear')
        refine3_hl_0 = F.relu(self.refine3_hl(torch.cat((refine4_hl_1, down3), 1)) + refine4_hl_1, True)
        refine3_hl_0 = (1 + self.attention3_hl(torch.cat((refine4_hl_1, down3), 1))) * refine3_hl_0
        refine3_hl_1 = F.relu(self.refine3_hl(torch.cat((refine3_hl_0, down3), 1)) + refine3_hl_0, True)
        refine3_hl_1_feature = (1 + self.attention3_hl(torch.cat((refine3_hl_0, down3), 1))) * refine3_hl_1

        refine3_hl_1 = F.upsample(refine3_hl_1_feature, size=down2.size()[2:], mode='bilinear')
        refine2_hl_0 = F.relu(self.refine2_hl(torch.cat((refine3_hl_1, down2), 1)) + refine3_hl_1, True)
        refine2_hl_0 = (1 + self.attention2_hl(torch.cat((refine3_hl_1, down2), 1))) * refine2_hl_0
        refine2_hl_1 = F.relu(self.refine2_hl(torch.cat((refine2_hl_0, down2), 1)) + refine2_hl_0, True)
        refine2_hl_1_feature = (1 + self.attention2_hl(torch.cat((refine2_hl_0, down2), 1))) * refine2_hl_1

        refine2_hl_1 = F.upsample(refine2_hl_1_feature, size=down1.size()[2:], mode='bilinear')
        refine1_hl_0 = F.relu(self.refine1_hl(torch.cat((refine2_hl_1, down1), 1)) + refine2_hl_1, True)
        refine1_hl_0 = (1 + self.attention1_hl(torch.cat((refine2_hl_1, down1), 1))) * refine1_hl_0
        refine1_hl_1 = F.relu(self.refine1_hl(torch.cat((refine1_hl_0, down1), 1)) + refine1_hl_0, True)
        refine1_hl_1_feature = (1 + self.attention1_hl(torch.cat((refine1_hl_0, down1), 1))) * refine1_hl_1

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        refine1_lh_0 = F.relu(self.refine1_lh(torch.cat((down1, down2), 1)) + down1, True)
        refine1_lh_0 = (1 + self.attention1_lh(torch.cat((down1, down2), 1))) * refine1_lh_0
        refine1_lh_1 = F.relu(self.refine1_lh(torch.cat((refine1_lh_0, down2), 1)) + refine1_lh_0, True)
        refine1_lh_1_feature = (1 + self.attention1_lh(torch.cat((refine1_lh_0, down2), 1))) * refine1_lh_1
        
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        refine2_lh_0 = F.relu(self.refine2_lh(torch.cat((refine1_lh_1_feature, down3), 1)) + refine1_lh_1_feature, True)
        refine2_lh_0 = (1 + self.attention2_lh(torch.cat((refine1_lh_1_feature, down3), 1))) * refine2_lh_0
        refine2_lh_1 = F.relu(self.refine2_lh(torch.cat((refine2_lh_0, down3), 1)) + refine2_lh_0, True)
        refine2_lh_1_feature = (1 + self.attention2_lh(torch.cat((refine2_lh_0, down3), 1))) * refine2_lh_1

        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')
        refine3_lh_0 = F.relu(self.refine3_lh(torch.cat((refine2_lh_1_feature, down3), 1)) + refine2_lh_1_feature, True)
        refine3_lh_0 = (1 + self.attention3_lh(torch.cat((refine2_lh_1_feature, down3), 1))) * refine3_lh_0
        refine3_lh_1 = F.relu(self.refine3_lh(torch.cat((refine3_lh_0, down3), 1)) + refine3_lh_0, True)
        refine3_lh_1_feature = (1 + self.attention3_lh(torch.cat((refine3_lh_0, down3), 1))) * refine3_lh_1

        down5 = F.upsample(down5, size=down1.size()[2:], mode='bilinear')
        refine4_lh_0 = F.relu(self.refine4_lh(torch.cat((refine3_lh_1_feature, down4), 1)) + refine3_lh_1_feature, True)
        refine4_lh_0 = (1 + self.attention4_lh(torch.cat((refine3_lh_1_feature, down4), 1))) * refine4_lh_0
        refine4_lh_1 = F.relu(self.refine4_lh(torch.cat((refine4_lh_0, down4), 1)) + refine4_lh_0, True)
        refine4_lh_1_feature = (1 + self.attention4_lh(torch.cat((refine4_lh_0, down4), 1))) * refine4_lh_1

        up1 = self.up1(torch.cat((refine1_hl_1_feature, refine4_lh_1_feature), 1))

        refine3_lh_1_feature  = F.interpolate(refine3_lh_1_feature, size=refine2_hl_1_feature.size()[2:], mode="bilinear")
        up2 = self.up2(torch.cat((refine2_hl_1_feature, refine3_lh_1_feature), 1))

        refine2_lh_1_feature  = F.interpolate(refine2_lh_1_feature, size=refine3_hl_1_feature.size()[2:], mode="bilinear")
        up3 = self.up3(torch.cat((refine3_hl_1_feature, refine2_lh_1_feature), 1))

        refine1_lh_1_feature  = F.interpolate(refine1_lh_1_feature, size=refine4_hl_1_feature.size()[2:], mode="bilinear")
        up4 = self.up4(torch.cat((refine4_hl_1_feature, refine1_lh_1_feature), 1))

        features = dict()
        features['p2'] = fpn1
        features['p3'] = up1
        features['p4'] = up2
        features['p5'] = up3
        features['p6'] = up4
        return features

@META_ARCH_REGISTRY.register()
class LISARCNN(GeneralizedRCNN):

    def __init__(self,cfg):
        super(LISARCNN, self).__init__(cfg)
        self.BDRAR_BACKBONE = BDRAR_BACKBONE().cuda()
        self.association_proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape(), shadow_object_part= False)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape(), shadow_object_part= True)
        self.to(self.device)
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if "associations" in batched_inputs[0]:
            gt_associations = [x["associations"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = self.BDRAR_BACKBONE([features['p2'], features['p3'], features['p4'], features['p5'], features['p6']])

        if self.association_proposal_generator:
            association_proposals, association_losses, pre_features, pre_proposals = self.association_proposal_generator(images, features, gt_associations)
        
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images,features,gt_instances,pre_proposals)

        _, detector_losses = self.roi_heads(images, features, association_proposals, proposals, gt_associations, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(association_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = self.BDRAR_BACKBONE([features['p2'], features['p3'], features['p4'], features['p5'], features['p6']])

        if detected_instances is None:
            if self.association_proposal_generator:
                association_proposals, _, pre_features, pre_proposals = self.association_proposal_generator(images, features)
            else:
                assert "associations" in batched_inputs[0]
                proposals = [x["associations"].to(self.device) for x in batched_inputs]
            if self.proposal_generator:
                # concat_features = {}
                # for pre_features,(k,v) in zip(pre_features,features.items()):
                #     concat_features[k] = torch.cat([v,pre_features],1)
                proposals, _ = self.proposal_generator(images,features,pre_proposals = pre_proposals)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results,associations, _ = self.roi_heads(images, features, association_proposals, proposals, None, None)
        
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r.to(torch.device('cpu'))})

            processed_associations = []
            for results_per_image, input_per_image, image_size in zip(
                associations, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_associations.append({"instances": r.to(torch.device('cpu'))})
            
            for instances, associations in zip(processed_results, processed_associations):
                _instances, _associations = matchor(instances["instances"],associations["instances"])
                _associations,_instances = combine_association(_instances,_associations)
                associations["instances"] = _associations
                instances["instances"] = _instances
                    

            return processed_results,processed_associations
        else:
            return results,associations


