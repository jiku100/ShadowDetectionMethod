import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
from utils.util import cal_subitizing
import cv2

def path_maker(root, edge = False):
  img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('jpg')]
  path_list = []
  if edge:
      for img_name in img_list:
          path_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                            os.path.join(root, 'ShadowMasks', img_name + '.png'),
                            os.path.join(root, 'EdgeMasks', img_name + '.png')))
  else:
      for img_name in img_list:
        path_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'ShadowMasks', img_name + '.png')))
  return path_list

def make_merge_src(src):
    B, G, R = cv2.split(src)
    B = np.where(B == 0, 1, B)
    G = np.where(G == 0, 1, G)
    R = np.where(R == 0, 1, R)
    s1 = np.log10(np.divide(G, R))
    s1 = np.uint8(cv2.normalize(s1, None, 0, 255, cv2.NORM_MINMAX))
    s2 = np.log10(np.divide(B, R))
    s2 = np.uint8(cv2.normalize(s2, None, 0, 255, cv2.NORM_MINMAX))
    s3 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    s4 = np.dstack((s1, s2, s3))
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    result1 = Image.fromarray(img).convert('RGB')
    result2 = Image.fromarray(s4).convert('RGB')
    return result1, result2

def make_first_src(src):
  img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
  return Image.fromarray(img).convert('RGB')

def make_second_src(src):
  # grayScale
  img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  result = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

  # log
  B, G, R = cv2.split(src)
  B = np.where(B == 0, 1, B)
  G = np.where(G == 0, 1, G)
  R = np.where(R == 0, 1, R)
  n1 = np.log10(np.divide(G, R))
  n2 = np.log10(np.divide(B, R))
  n3 = n1 + n2
  n3 = np.uint8(cv2.normalize(n3, None, 0, 255, cv2.NORM_MINMAX))
  result = cv2.cvtColor(n3, cv2.COLOR_GRAY2RGB)

  # smoothness
  img = cv2.bilateralFilter(src, 5, 15,10)
  img = cv2.pyrMeanShiftFiltering(img, 25, 25, 1)
  blur = cv2.GaussianBlur(img, (5,5), 1)
  smoothness = img - blur
  result = cv2.cvtColor(smoothness, cv2.COLOR_BGR2RGB)

  # gradient
  img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
  mag = cv2.magnitude(dx, dy)
  mag = np.clip(mag, 0, 255).astype(np.uint8)
  result = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
  return Image.fromarray(result).convert('RGB')

def make_target(target):
  target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
  return Image.fromarray(target).convert('L')


class SBU_EDGE_COUNT(data.Dataset):
  def __init__(self, root, joint_transform=None, src1_transform=None, src2_transform=None, target_transform=None, subitizing=False, subitizing_threshold=8, subitizing_min_size_per=0.005, edge=True):
    self.root = root
    self.joint_transform = joint_transform
    self.src1_transform = src1_transform
    self.src2_transform = src2_transform
    self.target_transform = target_transform
    self.paths = path_maker(root, edge)
    self.subitizing = subitizing
    # 8, 0.005
    self.subitizing_threshold = subitizing_threshold
    self.subitizing_min_size_per = subitizing_min_size_per

  def __getitem__(self, index):
      img_path, target_path, edge_path = self.paths[index]
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      target = cv2.imread(target_path, cv2.IMREAD_COLOR)
      edge = cv2.imread(edge_path, cv2.IMREAD_COLOR)
      src1, src2 = make_merge_src(img)
      target = make_target(target)
      edge = make_target(edge)
      count, percentage = cal_subitizing(target, threshold=self.subitizing_threshold,
                                              min_size_per=self.subitizing_min_size_per)
      count = torch.Tensor([count])
      src1, src2, target, edge = self.joint_transform(src1, src2, target, edge)
      src1 = self.src1_transform(src1)
      src2 = self.src2_transform(src2)
      target = self.target_transform(target)
      edge = self.target_transform(edge)
      sample = {'src1': src1, 'src2': src2, 'target': target, 'edge': edge, 'count': count}
      return sample

  def __len__(self):
    return len(self.paths)

class SBU_EDGE(data.Dataset):
  def __init__(self, root, joint_transform = None, src1_transform = None, src2_transform = None, target_transform = None, edge = True):
    self.root = root
    self.joint_transform = joint_transform
    self.src1_transform = src1_transform
    self.src2_transform = src2_transform
    self.target_transform = target_transform
    self.paths = path_maker(root, edge)

  def __getitem__(self, index):
      img_path, target_path, edge_path = self.paths[index]
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      target = cv2.imread(target_path, cv2.IMREAD_COLOR)
      edge = cv2.imread(edge_path, cv2.IMREAD_COLOR)
      src1, src2 = make_merge_src(img)
      target = make_target(target)
      edge = make_target(edge)
      src1, src2, target, edge = self.joint_transform(src1, src2, target, edge)
      src1 = self.src1_transform(src1)
      src2 = self.src2_transform(src2)
      target = self.target_transform(target)
      edge = self.target_transform(edge)
      sample = {'src1': src1, 'src2': src2, 'target': target, 'edge': edge}
      return sample

  def __len__(self):
    return len(self.paths)

class SBU_Merge(data.Dataset):
  def __init__(self, root, joint_transform = None, src1_transform = None, src2_transform = None, target_transform = None, edge = False):
    self.root = root
    self.joint_transform = joint_transform
    self.src1_transform = src1_transform
    self.src2_transform = src2_transform
    self.target_transform = target_transform
    self.paths = path_maker(root, edge)

  def __getitem__(self, index):
      img_path, target_path = self.paths[index]
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      target = cv2.imread(target_path, cv2.IMREAD_COLOR)
      src1, src2 = make_merge_src(img)
      target = make_target(target)
      src1, src2, target = self.joint_transform(src1, src2, target)
      src1 = self.src1_transform(src1)
      src2 = self.src2_transform(src2)
      target = self.target_transform(target)
      sample = {'src1': src1, 'src2': src2, 'target': target}
      return sample

  def __len__(self):
    return len(self.paths)

class SBU(data.Dataset):
  def __init__(self, root, joint_transform = None, src1_transform = None, src2_transform = None, target_transform = None, edge = False):
    self.root = root
    self.joint_transform = joint_transform
    self.src1_transform = src1_transform
    self.src2_transform = src2_transform
    self.target_transform = target_transform
    self.paths = path_maker(root, edge)

  def __getitem__(self, index):
      img_path, target_path = self.paths[index]
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      target = cv2.imread(target_path, cv2.IMREAD_COLOR)
      src1 = make_first_src(img)
      src2 = make_second_src(img)
      target = make_target(target)
      src1, src2, target = self.joint_transform(src1, src2, target)
      src1 = self.src1_transform(src1)
      src2 = self.src2_transform(src2)
      target = self.target_transform(target)
      sample = {'src1': src1, 'src2': src2, 'target': target}
      return sample

  def __len__(self):
    return len(self.paths)

def make_union_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowMasks')) if f.endswith('.png')]
    data_list = []

    for img_name in img_list:
        if img_name in label_list:
            data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                              os.path.join(root, 'ShadowMasks', img_name + '.png')))
        else:
            data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'), -1))

    return data_list


def make_labeled_dataset(root):
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowMasks')) if f.endswith('.png')]
    data_list = []
    for img_name in label_list:
        data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'ShadowMasks', img_name + '.png')))
    return data_list


class SBU_MTMT(data.Dataset):
    def __init__(self, root, joint_transform = None, src1_transform = None, src2_transform = None, target_transform = None):
        self.root = root
        self.imgs = make_union_dataset(root)
        self.joint_transform = joint_transform
        self.src1_transform = src1_transform
        self.src2_transform = src2_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, target_path = self.imgs[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if target_path == -1:
            target = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else:
            target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        src1 = make_first_src(img)
        src2 = make_second_src(img)
        target = make_target(target)
        src1, src2, target = self.joint_transform(src1, src2, target)
        src1 = self.src1_transform(src1)
        src2 = self.src2_transform(src2)
        target = self.target_transform(target)
        sample = {'src1': src1, 'src2': src2, 'target': target}

        return sample

    def __len__(self):
        return len(self.imgs)


def relabel_dataset(dataset):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, label = dataset.imgs[idx]
        if label == -1:
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs
