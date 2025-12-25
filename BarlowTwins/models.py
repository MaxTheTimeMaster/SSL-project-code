import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import MultiScaleRoIAlign


def _clean_resnet_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for p in ("module.", "encoder.", "backbone.", "model."):
            if k.startswith(p):
                k = k[len(p):]
        if k.startswith("fc."):
            continue
        if "projector" in k or k.startswith("head."):
            continue
        if k.endswith("num_batches_tracked"):
            continue
        out[k] = v
    return out


def load_checkpoint(module, bt_ckpt: str | None = "hub"):
    if bt_ckpt is None or bt_ckpt == "" or bt_ckpt == "hub":
        encoder = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
        sd = encoder.state_dict()
    else:
        if not os.path.exists(bt_ckpt):
            raise FileNotFoundError(bt_ckpt)
        ckpt = torch.load(bt_ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))

    sd = _clean_resnet_state_dict(sd)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    return missing, unexpected


class ResNetC4Backbone(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.out_channels = 1024

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return OrderedDict([("0", x)])


class Res5ROIHead(nn.Module):
    def __init__(self, layer4):
        super().__init__()
        self.res5 = layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.res5(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def make_detector(num_classes: int, det_backbone: str, bt_ckpt: str):
    det_backbone = str(det_backbone).lower()

    if det_backbone == "fpn":
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        missing, unexpected = load_checkpoint(model.backbone.body, bt_ckpt)
        model.det_backbone = "fpn"
        return model, missing, unexpected

    if det_backbone == "c4":
        resnet = torchvision.models.resnet50(weights=None)
        missing, unexpected = load_checkpoint(resnet, bt_ckpt)

        backbone = ResNetC4Backbone(resnet)

        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)
        box_head = Res5ROIHead(resnet.layer4)
        box_predictor = FastRCNNPredictor(2048, num_classes)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

        model = FasterRCNN(
            backbone,
            num_classes=None,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
        )
        model.det_backbone = "c4"
        return model, missing, unexpected

    raise ValueError(det_backbone)
