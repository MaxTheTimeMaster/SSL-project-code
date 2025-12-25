import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np

class DetTransform:
    def __init__(self, train: bool):
        self.train = train

    def __call__(self, img, target):
        if self.train and torch.rand(()) < 0.5:
            w, _ = img.size
            img = TF.hflip(img)
            boxes = target["boxes"]
            if boxes.numel():
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                target["boxes"] = torch.stack([xmin, boxes[:, 1], xmax, boxes[:, 3]], dim=1)

        img = TF.to_tensor(img)
        return img, target


class SegTransform:
    def __init__(
        self,
        train: bool,
        crop_size: int = 512,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        ignore_index: int = 255,
    ):
        self.train = train
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ignore_index = ignore_index
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def _pad_if_needed(self, img, mask, th, tw):
        w, h = img.size
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)
        return img, mask

    def __call__(self, img, mask):
        if self.train:
            scale = float(torch.empty(1).uniform_(self.min_scale, self.max_scale).item())
            w, h = img.size
            nh, nw = int(round(h * scale)), int(round(w * scale))
            img = TF.resize(img, [nh, nw], interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [nh, nw], interpolation=InterpolationMode.NEAREST)

            img, mask = self._pad_if_needed(img, mask, self.crop_size, self.crop_size)

            i, j, th, tw = torch.randint(
                0, img.size[1] - self.crop_size + 1, (1,)
            ).item(), torch.randint(0, img.size[0] - self.crop_size + 1, (1,)).item(), self.crop_size, self.crop_size

            img = TF.crop(img, i, j, th, tw)
            mask = TF.crop(mask, i, j, th, tw)

            if torch.rand(()) < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


def seg_pad_collate(batch, ignore_index: int = 255):
    imgs, masks = zip(*batch)
    max_h = max(x.shape[1] for x in imgs)
    max_w = max(x.shape[2] for x in imgs)

    b = len(imgs)
    out_imgs = imgs[0].new_zeros((b, 3, max_h, max_w))
    out_masks = masks[0].new_full((b, max_h, max_w), fill_value=ignore_index)

    for i, (im, ma) in enumerate(zip(imgs, masks)):
        _, h, w = im.shape
        out_imgs[i, :, :h, :w] = im
        out_masks[i, :h, :w] = ma

    return out_imgs, out_masks
