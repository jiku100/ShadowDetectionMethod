import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from .resnext.resnext101_5out import ResNeXt101


class EFMaker(nn.Module):
    def __init__(self):
        super(EFMaker, self).__init__()
        up0 = []
        up0.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        up0.append(nn.Sequential(nn.Conv2d(256, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        up0.append(nn.Sequential(nn.Conv2d(512, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        up0.append(nn.Sequential(nn.Conv2d(1024, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        up0.append(nn.Sequential(nn.Conv2d(2048, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class SCMaker(nn.Module):
    def __init__(self):
        super(SCMaker, self).__init__()

        self.number_per_fc = nn.Linear(64, 1)
        torch.nn.init.constant_(self.number_per_fc.weight, 0)

    def forward(self, x):
        vector = F.adaptive_avg_pool2d(x, output_size=1)
        vector = vector.view(vector.size(0), -1)
        sc = self.number_per_fc(vector)
        return sc


class DFMaker(nn.Module):
    def __init__(self):
        super(DFMaker, self).__init__()
        trans, up, DSS = [], [], []

        up.append(nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))

        up.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        up.append(nn.Sequential(nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        up.append(nn.Sequential(nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        up.append(nn.Sequential(nn.Conv2d(64, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        DSS.append(nn.Sequential(nn.Conv2d(128, 64, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        DSS.append(nn.Sequential(nn.Conv2d(192, 64, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        DSS.append(nn.Sequential(nn.Conv2d(256, 64, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))

        trans.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True)))

        self.up = nn.ModuleList(up)
        self.DSS = nn.ModuleList(DSS)
        self.trans = nn.ModuleList(trans)

    def forward(self, EFs, x_size=(416, 416)):
        DFs, U_tmp = [], []
        DF5 = self.up[4](EFs[4])
        DFs.append(DF5)
        U_tmp.append(DF5)

        EF5_2_EF4 = F.interpolate(DF5, EFs[3].size()[2:], mode='bilinear', align_corners=True)
        DF4_U = self.DSS[0](torch.cat([EF5_2_EF4, EFs[3]], dim=1))
        U_tmp.append(DF4_U)
        DF4 = self.up[3](DF4_U)
        DFs.insert(0, DF4)

        EF4_2_EF3 = F.interpolate(U_tmp[1], EFs[2].size()[2:], mode='bilinear', align_corners=True)
        EF5_2_EF3 = F.interpolate(U_tmp[0], EFs[2].size()[2:], mode='bilinear', align_corners=True)
        DF3_U = self.DSS[1](torch.cat([EF5_2_EF3, EF4_2_EF3, EFs[2]], dim=1))
        U_tmp.append(DF3_U)
        DF3 = self.up[2](DF3_U)
        DFs.insert(0, DF3)

        EF3_2_EF2 = F.interpolate(U_tmp[2], EFs[1].size()[2:], mode='bilinear', align_corners=True)
        EF4_2_EF2 = F.interpolate(U_tmp[1], EFs[1].size()[2:], mode='bilinear', align_corners=True)
        EF5_2_EF2 = F.interpolate(U_tmp[0], EFs[1].size()[2:], mode='bilinear', align_corners=True)
        DF2_U = self.DSS[2](torch.cat([EF5_2_EF2, EF4_2_EF2, EF3_2_EF2, EFs[1]], dim=1))
        U_tmp.append(DF2_U)
        DF2 = self.up[1](DF2_U)
        DFs.insert(0, DF2)

        DF1_U = EFs[0] + F.interpolate((self.trans[-1](DFs[-1])), EFs[0].size()[2:], mode='bilinear',
                                       align_corners=True)
        DF1 = self.up[0](DF1_U)
        DFs.insert(0, DF1)
        return DFs


class EdgeMaker(nn.Module):
    def __init__(self):
        super(EdgeMaker, self).__init__()
        self.edge_score = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        edge_mask = F.interpolate(self.edge_score(x), (416, 416), mode='bilinear', align_corners=True)
        return edge_mask


class RFMaker(nn.Module):
    def __init__(self):
        super(RFMaker, self).__init__()

        trans, up, score = [], [], []
        tmp, tmp_up = [], []
        tmp.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        tmp.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        tmp.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        tmp.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))

        tmp_up.append(nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        tmp_up.append(nn.Sequential(nn.Conv2d(32, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        tmp_up.append(nn.Sequential(nn.Conv2d(32, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        tmp_up.append(nn.Sequential(nn.Conv2d(32, 32, 7, 1, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 7, 1, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 7, 1, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))

        trans.append(nn.ModuleList(tmp))
        up.append(nn.ModuleList(tmp_up))

        self.trans, self.up = nn.ModuleList(trans), nn.ModuleList(up)
        self.relu = nn.ReLU()

    def forward(self, DFs, x_size=(416, 416)):
        RFs = []
        RF2_U = F.interpolate(self.trans[0][0](DFs[1]), DFs[0].size()[2:], mode='bilinear', align_corners=True) + DFs[0]
        RF2 = self.up[0][0](RF2_U)
        RFs.insert(0, RF2)

        RF3_U = F.interpolate(self.trans[0][1](DFs[2]), DFs[0].size()[2:], mode='bilinear', align_corners=True) + DFs[0]
        RF3 = self.up[0][1](RF3_U)
        RFs.insert(0, RF3)

        RF4_U = F.interpolate(self.trans[0][2](DFs[3]), DFs[0].size()[2:], mode='bilinear', align_corners=True) + DFs[0]
        RF4 = self.up[0][2](RF4_U)
        RFs.insert(0, RF4)

        RF5_U = F.interpolate(self.trans[0][3](DFs[4]), DFs[0].size()[2:], mode='bilinear', align_corners=True) + DFs[0]
        RF5 = self.up[0][3](RF5_U)
        RFs.insert(0, RF5)

        Final_feature = RFs[0]
        for i_fea in range(len(RFs) - 1):
            Final_feature = self.relu(torch.add(Final_feature, F.interpolate((RFs[i_fea + 1]), RFs[0].size()[2:],
                                                                             mode='bilinear', align_corners=True)))

        return RFs, Final_feature


class ShadowMaker(nn.Module):
    def __init__(self):
        super(ShadowMaker, self).__init__()
        self.DF_Shadow_Maker = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(16, 1, 1)
        )
        self.RF_Shadow_Maker = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )

    def forward(self, DFs, RFs, Final_feature, x_size=(416, 416)):
        sub_mask = []
        sub_mask.append(F.interpolate(self.DF_Shadow_Maker(DFs[1]), x_size, mode='bilinear', align_corners=True))
        sub_mask.append(F.interpolate(self.DF_Shadow_Maker(DFs[2]), x_size, mode='bilinear', align_corners=True))
        sub_mask.append(F.interpolate(self.DF_Shadow_Maker(DFs[3]), x_size, mode='bilinear', align_corners=True))
        sub_mask.append(F.interpolate(self.DF_Shadow_Maker(DFs[4]), x_size, mode='bilinear', align_corners=True))

        sub_mask.append(F.interpolate(self.RF_Shadow_Maker(RFs[0]), x_size, mode='bilinear', align_corners=True))
        sub_mask.append(F.interpolate(self.RF_Shadow_Maker(RFs[1]), x_size, mode='bilinear', align_corners=True))
        sub_mask.append(F.interpolate(self.RF_Shadow_Maker(RFs[2]), x_size, mode='bilinear', align_corners=True))
        sub_mask.append(F.interpolate(self.RF_Shadow_Maker(RFs[3]), x_size, mode='bilinear', align_corners=True))

        final_mask = F.interpolate(self.RF_Shadow_Maker(Final_feature), x_size, mode='bilinear', align_corners=True)

        return sub_mask, final_mask


class MTMT(nn.Module):
    def __init__(self):
        super(MTMT, self).__init__()
        self.resnext = ResNeXt101()
        self.efMaker = EFMaker()
        self.scMaker = SCMaker()
        self.dfMaker = DFMaker()
        self.edgeMaker = EdgeMaker()
        self.rfMaker = RFMaker()
        self.shadowMaker = ShadowMaker()

    def forward(self, x):
        res = self.resnext(x)
        EF_layers = self.efMaker(res)
        DF_layers = self.dfMaker(EF_layers)
        Shadow_count = self.scMaker(DF_layers[-1])
        Edge_mask = self.edgeMaker(DF_layers[0])
        RF_layers, Final_feature = self.rfMaker(DF_layers)
        sub_mask, final_mask = self.shadowMaker(DF_layers, RF_layers, Final_feature)
        return Edge_mask, sub_mask, Shadow_count, final_mask