
import torch
from voc_to_target import voc_det_to_target
from transforms import DetTransform
from torchvision.datasets import VOCSegmentation, VOCDetection
import torchvision.transforms.functional as TF
from voc_to_target import voc_det_to_eval_target

def det_collate(batch):
    return tuple(zip(*batch))

class VOCDet(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set, train, keep_difficult=False):
        self.ds = VOCDetection(root=root, year=year, image_set=image_set, download=False)
        self.tf = DetTransform(train=train)
        self.keep_difficult = keep_difficult

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, voc_target = self.ds[idx]
        target = voc_det_to_target(voc_target, image_id=idx, keep_difficult=self.keep_difficult)
        img, target = self.tf(img, target)
        return img, target

class VOCDetEval(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set):
        self.ds = VOCDetection(root=root, year=year, image_set=image_set, download=False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, voc_target = self.ds[idx]
        img = TF.to_tensor(img)
        target = voc_det_to_eval_target(voc_target, image_id=idx)
        return img, target
