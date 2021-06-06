import random
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, src1, src2, target, edge):
        for t in self.transforms:
            src1, src2, target, edge = t(src1, src2, target, edge)
        return src1, src2, target, edge


class RandomHorizontallyFlip(object):
    def __call__(self, src1, src2, target, edge):
        if random.random() < 0.5:
            return src1.transpose(Image.FLIP_LEFT_RIGHT), src2.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT)
        return src1, src2, target, edge


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, src1, src2, target, edge):
        return src1.resize(self.size, Image.BILINEAR), src2.resize(self.size, Image.NEAREST), target.resize(self.size, Image.NEAREST), edge.resize(self.size, Image.NEAREST)