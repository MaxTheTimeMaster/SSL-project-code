import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset

from models import make_detector
from voc_detection_dataset import VOCDet, VOCDetEval, det_collate
from voc07_eval import evaluate_voc07_map50, evaluate_coco_style, evaluate_seg_miou
from logger import Logger
from tqdm.auto import tqdm
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def _infinite(loader):
    while True:
        for b in loader:
            yield b


def _set_trainable_backbone(model, trainable_backbone_layers: int, train_fpn: int):
    if getattr(model, "det_backbone", "fpn") == "fpn":
        body = model.backbone.body

        for p in body.parameters():
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
            m = getattr(body, name)
            for p in m.parameters():
                p.requires_grad = True

        for p in model.backbone.fpn.parameters():
            p.requires_grad = bool(train_fpn)

        return

    if getattr(model, "det_backbone", None) == "c4":
        resnet = model.backbone.resnet

        for p in resnet.parameters():
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
            m = getattr(resnet, name)
            for p in m.parameters():
                p.requires_grad = True

        return



def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _step_lr(opt, step: int, milestones: list[int], gamma: float):
    if step in set(int(x) for x in milestones):
        for pg in opt.param_groups:
            pg["lr"] *= float(gamma)


def train_det(args):
    torch.manual_seed(int(getattr(args, "seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(getattr(args, "seed", 0)))

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(str(out_dir))
    accum_steps = int(getattr(args, "accum_steps", 1))
    clip_grad = float(getattr(args, "clip_grad_norm", 0.0))

    
    print(f"[init] device: {device}")
    print(f"[init] out_dir: {out_dir}")

    model, missing, unexpected = make_detector(
        num_classes=21,
        det_backbone=getattr(args, "det_backbone", "fpn"),
        bt_ckpt=getattr(args, "bt_ckpt", "hub"),
    )
    model = model.to(device)

    # linear-probe / last-k layers
    _set_trainable_backbone(
        model,
        trainable_backbone_layers=int(getattr(args, "trainable_backbone_layers", 5)),
        train_fpn=int(getattr(args, "train_fpn", 1)),
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] trainable params: {n_trainable / 1e6:.2f}M")

    log.log(0, {
        "train/params_trainable": float(n_trainable),
        "load/missing_keys": float(len(missing)),
        "load/unexpected_keys": float(len(unexpected)),
    })

    # VOC07+12 trainval
    print("[data] loading VOC 2007 trainval...")
    ds07 = VOCDet(root=args.data_root, year="2007", image_set="trainval", train=True, keep_difficult=False)
    print(f"[data] VOC2007 trainval size: {len(ds07)}")

    print("[data] loading VOC 2012 trainval...")
    ds12 = VOCDet(root=args.data_root, year="2012", image_set="trainval", train=True, keep_difficult=False)
    print(f"[data] VOC2012 trainval size: {len(ds12)}")

    train_ds = ConcatDataset([ds07, ds12])
    print(f"[data] total train size: {len(train_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=det_collate,
        drop_last=True,
        persistent_workers=True
    )
    it = _infinite(train_dl)

    # VOC07 test eval (VOC07 metric needs difficult)
    print("[data] loading VOC 2007 test for eval...")
    eval_ds = VOCDetEval(root=args.data_root, year="2007", image_set="test")
    print(f"[data] VOC2007 test size: {len(eval_ds)}")

    eval_dl = DataLoader(
        eval_ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=det_collate,
    )

    opt = torch.optim.SGD(_trainable_params(model), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler(enabled=bool(int(getattr(args, "amp", 1))) and torch.cuda.is_available())

    max_iters = int(getattr(args, "max_iters", 24000))
    lr_steps = list(getattr(args, "lr_steps", [18000, 22000]))
    gamma = float(getattr(args, "gamma", 0.1))
    log_every = int(getattr(args, "log_every", 20))
    eval_every = int(getattr(args, "eval_every", 2000))

    print(f"[train] max_iters={max_iters}, lr={args.lr}, lr_steps={lr_steps}, gamma={gamma}")
    print(f"[train] batch_size={args.batch_size}, num_workers={args.num_workers}")
    print(f"[train] log_every={log_every}, eval_every={eval_every}")
    print(f"[train] global_batch_size={args.batch_size * accum_steps}, clip_grad={clip_grad}")
    if scaler.is_enabled():
        print("[train] AMP: enabled")
    else:
        print("[train] AMP: disabled")

    best = -1.0
    t0 = time.perf_counter()
    pbar = tqdm(range(1, max_iters + 1), desc="det-train", ncols=100)

    base_lr = float(args.lr)
    warmup_iters = 1000
    warmup_factor = 0.333

    milestones = [int(x) for x in lr_steps]
    milestones.sort()

    def lr_at(step: int):
        lr = base_lr
        if step <= warmup_iters:
            alpha = step / warmup_iters
            lr = base_lr * (warmup_factor + alpha * (1.0 - warmup_factor))
        for m in milestones:
            if step >= m:
                lr *= gamma
        return lr

    def set_lr(lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

    for step in pbar:
        set_lr(lr_at(step))
        model.train()

        opt.zero_grad(set_to_none=True)

        loss_total = 0.0
        losses_total = {}
        for micro in range(accum_steps):
            imgs, targets = next(it)
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with autocast(enabled=scaler.is_enabled()):
                losses = model(imgs, targets)
                loss = sum(losses.values()) / accum_steps

            scaler.scale(loss).backward()

            loss_total += float(loss.item())
            for k, v in losses.items():
                losses_total[k] = losses_total.get(k, 0.0) + float(v.item()) / accum_steps

        if clip_grad > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=clip_grad)

        scaler.step(opt)
        scaler.update()


        if step % log_every == 0:
            dt = (time.perf_counter() - t0) / step
            scalars = {
                "train/loss": float(loss_total),
                "train/lr": float(opt.param_groups[0]["lr"]),
                "time/iter_s": float(dt),
            }
            for k, v in losses_total.items():
                scalars[f"train/{k}"] = float(v)
            if torch.cuda.is_available():
                scalars["gpu/max_mem_gb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            log.log(step, scalars)

            # обновляем postfix у tqdm
            pbar.set_postfix({
                "loss": scalars["train/loss"],
                "lr": scalars["train/lr"],
                "t/it_s": round(scalars["time/iter_s"], 4),
            })

        if step % eval_every == 0 or step == max_iters:
            print(f"\n[eval] step {step}: running VOC07 mAP@0.5 eval...")
            map50 = evaluate_voc07_map50(model, eval_dl, device) * 100
            res = evaluate_coco_style(model, eval_dl, device, score_thresh=0.05)
            log.log(step, {
                "val/APall": 100.0 * res["map"],
                "val/AP50":  100.0 * res["map_50"],
                "val/AP75":  100.0 * res["map_75"],
            })
            # miou, pixacc = evaluate_seg_miou(model, eval_dl, device)
            # log.log(step, {"val/mIoU": 100.0 * miou, "val/pixAcc": 100.0 * pixacc})

            print(f"[eval] step {step}: mAP50_voc07 = {map50:.4f}, best so far = {best:.4f}")

            log.log(step, {"val/mAP50_voc07": float(map50)})

            if map50 > best:
                best = float(map50)
                ckpt_path = out_dir / "det_best.pth"
                torch.save({"model": model.state_dict(), "step": step, "mAP50_voc07": best}, ckpt_path)
                print(f"[ckpt] new best model saved at step {step} to {ckpt_path}")

    last_path = out_dir / "det_last.pth"
    torch.save({"model": model.state_dict(), "step": max_iters}, last_path)
    print(f"[ckpt] last model saved to {last_path}")
    log.close()
    print("[done] training finished.")
