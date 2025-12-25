from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation, SBDataset
from transforms import SegTransform

def _stem(p):
    return Path(p).stem

class VOCAugSeg(Dataset):
    def __init__(self, root: str, split: str, crop_size: int = 512):
        assert split in ("trainaug", "train", "val")

        self.root = root
        self.split = split

        if split == "val":
            self.ds = VOCSegmentation(root=root, year="2012", image_set="val", download=False,
                                     transforms=SegTransform(train=False, crop_size=crop_size))
            self.samples = None
            return

        if split == "train":
            self.ds = VOCSegmentation(root=root, year="2012", image_set="train", download=False,
                                     transforms=SegTransform(train=True, crop_size=crop_size))
            self.samples = None
            return

        voc_train = VOCSegmentation(root=root, year="2012", image_set="train", download=False)
        voc_val = VOCSegmentation(root=root, year="2012", image_set="val", download=False)
        sbd_tr = SBDataset(root=root, image_set="train", mode="segmentation", download=False)
        sbd_va = SBDataset(root=root, image_set="val", mode="segmentation", download=False)

        voc_val_ids = set(_stem(p) for p in voc_val.images)

        sbd_map = {}
        for ds in (sbd_tr, sbd_va):
            for i, p in enumerate(ds.images):
                sid = _stem(p)
                if sid in voc_val_ids:
                    continue
                if sid not in sbd_map:
                    sbd_map[sid] = (ds, i)

        voc_train_map = { _stem(p): i for i, p in enumerate(voc_train.images) }

        final_ids = sorted(set(sbd_map.keys()) | set(voc_train_map.keys()))
        samples = []
        for sid in final_ids:
            if sid in sbd_map:
                samples.append(("sbd", sid))
            else:
                samples.append(("voc", sid))

        self.voc_train = voc_train
        self.sbd_tr = sbd_tr
        self.sbd_va = sbd_va
        self.sbd_map = sbd_map
        self.voc_train_map = voc_train_map
        self.samples = samples
        self.tf = SegTransform(train=True, crop_size=crop_size)

    def __len__(self):
        if self.samples is None:
            return len(self.ds)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.samples is None:
            return self.ds[idx]

        src, sid = self.samples[idx]
        if src == "sbd":
            ds, j = self.sbd_map[sid]
            img, mask = ds[j]
        else:
            j = self.voc_train_map[sid]
            img, mask = self.voc_train[j]

        return self.tf(img, mask)
