import torch

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]
VOC_CLS2ID = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}


def voc_det_to_target(voc_target, image_id, keep_difficult=False):
    ann = voc_target["annotation"]
    objs = ann.get("object", [])
    if isinstance(objs, dict):
        objs = [objs]

    boxes, labels = [], []
    for obj in objs:
        if (not keep_difficult) and obj.get("difficult", "0") == "1":
            continue
        name = obj["name"]
        if name not in VOC_CLS2ID:
            continue
        b = obj["bndbox"]
        xmin = float(b["xmin"]) - 1.0
        ymin = float(b["ymin"]) - 1.0
        xmax = float(b["xmax"]) - 1.0
        ymax = float(b["ymax"]) - 1.0
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(VOC_CLS2ID[name])

    if boxes:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        area = torch.zeros((0,), dtype=torch.float32)

    target = {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": area,
        "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
    }
    return target

def voc_det_to_eval_target(voc_target, image_id):
    ann = voc_target["annotation"]
    objs = ann.get("object", [])
    if isinstance(objs, dict):
        objs = [objs]

    boxes, labels, difficult = [], [], []
    for obj in objs:
        name = obj["name"]
        if name not in VOC_CLS2ID:
            continue
        b = obj["bndbox"]
        xmin = float(b["xmin"]) - 1.0
        ymin = float(b["ymin"]) - 1.0
        xmax = float(b["xmax"]) - 1.0
        ymax = float(b["ymax"]) - 1.0
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(VOC_CLS2ID[name])
        difficult.append(obj.get("difficult", "0") == "1")

    if boxes:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        difficult = torch.tensor(difficult, dtype=torch.bool)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        difficult = torch.zeros((0,), dtype=torch.bool)
        area = torch.zeros((0,), dtype=torch.float32)

    return {
        "boxes": boxes,
        "labels": labels,
        "difficult": difficult,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": area,
        "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
    }
