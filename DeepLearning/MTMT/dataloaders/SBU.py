import os
import os.path

import torch.utils.data as data
from PIL import Image
import torch
from utils.util import cal_subitizing
import matplotlib.pyplot as plt

NO_LABEL = -1


def make_union_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowMasks')) if f.endswith('.png')]
    data_list = []
    if edge:
        edge_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'EdgeMasks')) if f.endswith('.png')]
        for img_name in img_list:
            if img_name in label_list:  # filter labeled data by seg label
                # if img_name in edge_list: # filter labeled data by edge label
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                                  os.path.join(root, 'ShadowMasks', img_name + '.png'),
                                  os.path.join(root, 'EdgeMasks', img_name + '.png')))
            else:  # we set label=-1 when comes to unlebaled data
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'), -1, -1))
    else:
        for img_name in img_list:
            if img_name in label_list:  # filter labeled data by seg label
                # if img_name in edge_list:  # filter labeled data by edge label
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                                  os.path.join(root, 'ShadowMasks', img_name + '.png')))
            else:  # we set label=-1 when comes to unlebaled data
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'), -1))

    return data_list


def make_labeled_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowMasks')) if f.endswith('.png')]
    data_list = []
    if edge:
        # for img_name in img_list:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                              os.path.join(root, 'ShadowMasks', img_name + '.png'),
                              os.path.join(root, 'EdgeMasks', img_name + '.png')))
    else:
        # for img_name in img_list:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                              os.path.join(root, 'ShadowMasks', img_name + '.png')))
    return data_list


class SBU(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, mod='union', multi_task=True, subitizing=False,
                 subitizing_threshold=8, subitizing_min_size_per=0.005, edge=False):
        assert (mod in ['union', 'labeled'])
        self.root = root
        self.mod = mod
        self.multi_task = multi_task
        if self.mod == 'union':
            self.imgs = make_union_dataset(root, edge)
        else:
            self.imgs = make_labeled_dataset(root, edge)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.edge = edge
        self.subitizing = subitizing
        # 8, 0.005
        self.subitizing_threshold = subitizing_threshold
        self.subitizing_min_size_per = subitizing_min_size_per

    def __getitem__(self, index):
        img_path, gt_path, edge_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if gt_path == -1:  # unlabeled data
            target = Image.new('L', (img.size[0], img.size[1]))
            edge = Image.new('L', (img.size[0], img.size[1]))
            img, target, edge = self.joint_transform(img, target, edge)
            img = self.transform(img)
            target = self.target_transform(target)
            edge = self.target_transform(edge)
            number_per = torch.ones(1)
            sample = {'image': img, 'label': target, 'number_per': number_per, 'edge': edge}

        else:  # labeled data
            target = Image.open(gt_path).convert('L')
            edge = Image.open(edge_path).convert('L')
            number_per, percentage = cal_subitizing(target, threshold=self.subitizing_threshold,
                                                    min_size_per=self.subitizing_min_size_per)
            number_per = torch.Tensor([number_per])
            img, target, edge = self.joint_transform(img, target, edge)
            img = self.transform(img)
            target = self.target_transform(target)
            edge = self.target_transform(edge)
            sample = {'image': img, 'label': target, 'number_per': number_per, 'edge': edge}

        return sample

    def __len__(self):
        return len(self.imgs)


def relabel_dataset(dataset, edge_able=False):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        if not edge_able:
            path, label = dataset.imgs[idx]
        else:
            path, label, edge = dataset.imgs[idx]
        if label == -1:
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs