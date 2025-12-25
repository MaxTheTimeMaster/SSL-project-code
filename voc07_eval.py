import numpy as np
import torch

def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-12)

def _voc07_ap(rec, prec):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return float(ap)

@torch.no_grad()
def evaluate_voc07_map50(model, loader, device, num_classes=21, iou_thresh=0.5, max_dets=100):
    model.eval()

    gt = {c: {} for c in range(1, num_classes)}
    npos = {c: 0 for c in range(1, num_classes)}
    dets = {c: [] for c in range(1, num_classes)}

    for imgs, targets in loader:
        imgs = [x.to(device, non_blocking=True) for x in imgs]
        outs = model(imgs)

        for out, t in zip(outs, targets):
            img_id = int(t["image_id"].item()) if torch.is_tensor(t["image_id"]) else int(t["image_id"])
            tboxes = t["boxes"].float()
            tlabels = t["labels"].long()
            tdif = t["difficult"].bool()

            for c in range(1, num_classes):
                m = tlabels == c
                if m.any():
                    b = tboxes[m].cpu()
                    d = tdif[m].cpu()
                    gt[c][img_id] = {
                        "boxes": b,
                        "difficult": d,
                        "det": torch.zeros((b.shape[0],), dtype=torch.bool),
                    }
                    npos[c] += int((~d).sum().item())

            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            if max_dets is not None and scores.numel() > max_dets:
                keep = torch.topk(scores, k=max_dets).indices
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            for c in range(1, num_classes):
                mc = labels == c
                if mc.any():
                    bc = boxes[mc]
                    sc = scores[mc]
                    for j in range(bc.shape[0]):
                        dets[c].append((img_id, float(sc[j].item()), bc[j]))

    aps = []
    for c in range(1, num_classes):
        preds = dets[c]
        if len(preds) == 0:
            aps.append(0.0)
            continue

        preds.sort(key=lambda x: -x[1])
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for k, (img_id, score, box) in enumerate(preds):
            if img_id not in gt[c]:
                fp[k] = 1.0
                continue
            g = gt[c][img_id]
            gboxes = g["boxes"]
            gdif = g["difficult"]
            gdet = g["det"]

            ious = _box_iou(box[None, :].float(), gboxes.float())[0]
            j = int(torch.argmax(ious).item()) if ious.numel() else -1
            ov = float(ious[j].item()) if j >= 0 else 0.0

            if ov >= iou_thresh:
                if gdif[j]:
                    tp[k] = 0.0
                    fp[k] = 0.0
                elif not bool(gdet[j].item()):
                    tp[k] = 1.0
                    gdet[j] = True
                else:
                    fp[k] = 1.0
            else:
                fp[k] = 1.0

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / max(float(npos[c]), 1.0)
        prec = tp / np.maximum(tp + fp, 1e-12)
        aps.append(_voc07_ap(rec, prec))

    return float(np.mean(aps))


from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def evaluate_coco_style(model, loader, device, score_thresh=None):
    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()

    old_thresh = model.roi_heads.score_thresh
    old_topk = model.roi_heads.detections_per_img

    if score_thresh is not None:
        model.roi_heads.score_thresh = float(score_thresh)
    model.roi_heads.detections_per_img = 100

    for imgs, targets in loader:
        imgs = [x.to(device, non_blocking=True) for x in imgs]
        preds = model(imgs)

        t2 = []
        for t in targets:
            keep = ~t["difficult"]
            t2.append({
                "boxes": t["boxes"][keep].to(device),
                "labels": t["labels"][keep].to(device),
            })

        metric.update(preds, t2)

    res = metric.compute()

    model.roi_heads.score_thresh = old_thresh
    model.roi_heads.detections_per_img = old_topk

    keys = [
        "map", "map_50", "map_75",
        "mar_1", "mar_10", "mar_100",
        "map_small", "map_medium", "map_large",
        "mar_small", "mar_medium", "mar_large",
    ]
    out = {}
    for k in keys:
        if k in res and isinstance(res[k], torch.Tensor) and res[k].numel() == 1:
            out[k] = float(res[k].detach().cpu().item())
    return out

from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

@torch.no_grad()
def evaluate_seg_miou(model, loader, device, num_classes=21, ignore_index=255):
    miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index).to(device)
    pix = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index, average="micro").to(device)

    model.eval()
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)["out"]
        pred = logits.argmax(1)
        miou.update(pred, masks)
        pix.update(pred, masks)

    return float(miou.compute().item()), float(pix.compute().item())

