import argparse
import warnings
warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["seg", "det"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--bt_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs_voc")
    p.add_argument("--device", type=str, default="cuda:3")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--max_iters", type=int, default=24000)
    p.add_argument("--lr_steps", type=int, nargs="+", default=[18000, 22000])
    p.add_argument("--gamma", type=float, default=0.1)

    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=2000)

    p.add_argument("--trainable_backbone_layers", type=int, default=5)
    p.add_argument("--train_fpn", type=int, default=1)

    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--det_backbone", choices=["fpn", "c4"], default="fpn")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--clip_grad_norm", type=float, default=0.0)



    args = p.parse_args()

    if args.task == "det":
        from train_detection import train_det
        train_det(args)
    else:
        from train_segmentation import train_seg
        train_seg(args)

if __name__ == "__main__":
    main()
