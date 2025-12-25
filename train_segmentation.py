import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models import make_segmenter
from voc_segmentation_dataset import VOCSeg, seg_pad_collate
from logger import Logger
from tqdm.auto import tqdm

from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _set_trainable_backbone_seg(model, trainable_backbone_layers: int):
    bb = model.backbone
    for p in bb.parameters():
        p.requires_grad = False

    layers = []
    if trainable_backbone_layers >= 1:
        layers.append("layer4")
    if trainable_backbone_layers >= 2:
        layers.append("layer3")
    if trainable_backbone_layers >= 3:
        layers.append("layer2")
    if trainable_backbone_layers >= 4:
        layers.append("layer1")
    if trainable_backbone_layers >= 5:
        layers.extend(["conv1", "bn1"])

    for name in layers:
        m = getattr(bb, name)
        for p in m.parameters():
            p.requires_grad = True


def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.inference_mode()
def evaluate_seg(model, loader, device, num_classes=21, ignore_index=255):
    miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index).to(device)
    pix = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index, average="micro").to(device)
    macc = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index, average="macro").to(device)

    model.eval()
    loss_sum = 0.0
    n = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out = model(imgs)
        logits = out["out"]
        loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)
        if "aux" in out:
            loss = loss + 0.4 * F.cross_entropy(out["aux"], masks, ignore_index=ignore_index)

        b = imgs.shape[0]
        loss_sum += float(loss.item()) * b
        n += b

        pred = logits.argmax(1)
        miou.update(pred, masks)
        pix.update(pred, masks)
        macc.update(pred, masks)

    return {
        "loss": loss_sum / max(n, 1),
        "mIoU": float(miou.compute().item()),
        "pixAcc": float(pix.compute().item()),
        "mAcc": float(macc.compute().item()),
    }


def train_seg(args):
    torch.manual_seed(int(getattr(args, "seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(getattr(args, "seed", 0)))

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(str(out_dir))

    accum_steps = int(getattr(args, "accum_steps", 1))
    clip_grad = float(getattr(args, "clip_grad_norm", 0.0))

    seg_arch = getattr(args, "seg_arch", "fcn")
    seg_split = getattr(args, "seg_split", "trainaug")
    seg_crop = int(getattr(args, "seg_crop", 512))
    ignore_index = int(getattr(args, "ignore_index", 255))
    aux_loss = bool(int(getattr(args, "aux_loss", 1)))

    model, missing, unexpected = make_segmenter(
        num_classes=21,
        seg_arch=seg_arch,
        bt_ckpt=getattr(args, "bt_ckpt", "hub"),
        aux_loss=aux_loss,
    )
    model = model.to(device)

    _set_trainable_backbone_seg(model, int(getattr(args, "trainable_backbone_layers", 5)))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.log(0, {
        "train/params_trainable": float(n_trainable),
        "load/missing_keys": float(len(missing)),
        "load/unexpected_keys": float(len(unexpected)),
    })

    train_ds = VOCSeg(root=args.data_root, split=seg_split, crop_size=seg_crop, ignore_index=ignore_index)
    val_ds = VOCSeg(root=args.data_root, split="val", crop_size=seg_crop, ignore_index=ignore_index)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=lambda b: seg_pad_collate(b, ignore_index=ignore_index),
    )
    val_bs = int(getattr(args, "eval_batch_size", 8))
    val_dl = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: seg_pad_collate(b, ignore_index=ignore_index),
    )

    opt = torch.optim.SGD(_trainable_params(model), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler(enabled=bool(int(getattr(args, "amp", 1))) and torch.cuda.is_available())

    max_iters = int(getattr(args, "max_iters", 20000))
    log_every = int(getattr(args, "log_every", 20))
    eval_every = int(getattr(args, "eval_every", 1000))

    base_lr = float(args.lr)
    warmup_iters = int(getattr(args, "warmup_iters", 1000))
    warmup_factor = float(getattr(args, "warmup_factor", 0.333))
    lr_policy = str(getattr(args, "lr_policy", "poly")).lower()
    poly_power = float(getattr(args, "poly_power", 0.9))
    lr_steps = list(getattr(args, "lr_steps", [14000, 18000]))
    gamma = float(getattr(args, "gamma", 0.1))
    milestones = sorted(int(x) for x in lr_steps)

    def base_sched(step: int):
        if lr_policy == "step":
            lr = base_lr
            for m in milestones:
                if step >= m:
                    lr *= gamma
            return lr
        t = step / max(max_iters, 1)
        return base_lr * (1.0 - t) ** poly_power

    def lr_at(step: int):
        lr = base_sched(step)
        if step <= warmup_iters:
            alpha = step / max(warmup_iters, 1)
            scale = warmup_factor + alpha * (1.0 - warmup_factor)
            lr *= scale
        return lr

    def set_lr(lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

    best = -1.0
    t0 = time.perf_counter()
    pbar = tqdm(range(1, max_iters + 1), desc="seg-train", ncols=100)

    it = iter(train_dl)
    for step in pbar:
        set_lr(lr_at(step))
        model.train()
        opt.zero_grad(set_to_none=True)

        loss_total = 0.0
        loss_out_total = 0.0
        loss_aux_total = 0.0

        for micro in range(accum_steps):
            try:
                imgs, masks = next(it)
            except StopIteration:
                it = iter(train_dl)
                imgs, masks = next(it)

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(enabled=scaler.is_enabled()):
                out = model(imgs)
                logits = out["out"]
                loss_out = F.cross_entropy(logits, masks, ignore_index=ignore_index)
                loss_aux = 0.0
                if "aux" in out:
                    loss_aux = F.cross_entropy(out["aux"], masks, ignore_index=ignore_index)
                loss = (loss_out + 0.4 * loss_aux) / accum_steps

            scaler.scale(loss).backward()

            loss_total += float(loss.item())
            loss_out_total += float(loss_out.item()) / accum_steps
            loss_aux_total += float(loss_aux.item()) / accum_steps if isinstance(loss_aux, torch.Tensor) else 0.0

        if clip_grad > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=clip_grad)

        scaler.step(opt)
        scaler.update()

        if step % log_every == 0:
            dt = (time.perf_counter() - t0) / step
            scalars = {
                "train/loss": float(loss_total),
                "train/loss_out": float(loss_out_total),
                "train/loss_aux": float(loss_aux_total),
                "train/lr": float(opt.param_groups[0]["lr"]),
                "time/iter_s": float(dt),
            }
            log.log(step, scalars)
            pbar.set_postfix({"loss": scalars["train/loss"], "lr": scalars["train/lr"], "t/it_s": round(scalars["time/iter_s"], 4)})

        if step % eval_every == 0 or step == max_iters:
            res = evaluate_seg(model, val_dl, device, num_classes=21, ignore_index=ignore_index)
            log.log(step, {
                "val/loss": float(res["loss"]),
                "val/mIoU": 100.0 * float(res["mIoU"]),
                "val/pixAcc": 100.0 * float(res["pixAcc"]),
                "val/mAcc": 100.0 * float(res["mAcc"]),
            })

            miou = float(res["mIoU"])
            if miou > best:
                best = miou
                torch.save({"model": model.state_dict(), "step": step, "mIoU": best}, out_dir / "seg_best.pth")

    torch.save({"model": model.state_dict(), "step": max_iters}, out_dir / "seg_last.pth")
    log.close()
