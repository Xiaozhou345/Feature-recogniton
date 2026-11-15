"""统一训练框架（核心改动：原按 val loss 保存最优模型 → 改为按 val mIoU 保存）
对应《项目分工与执行规划》第四节的统一实验协议：
- 固定项：seed=42、梯度裁剪、早停；ReduceLROnPlateau 调度器可用 --no-scheduler 关闭（短训建议关闭，保持 lr 恒定以公平比较初始学习率）
- 训练策略改进（答辩素材）：验证集以 mIoU 选最优 checkpoint，与最终评测指标对齐
"""
import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models import build_model, load_sfpn_weights
from dataset import RemoteSensingDataset
from losses import build_loss
from metrics import SegmentationMetrics, CLASS_NAMES_7


def parse_args():
    p = argparse.ArgumentParser(description="遥感图像 7 类语义分割统一训练框架")
    # 数据
    p.add_argument("--train-img", required=True)
    p.add_argument("--train-mask", required=True)
    p.add_argument("--val-img", required=True)
    p.add_argument("--val-mask", required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    # 模型
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "mobilenet_v3", "vgg16_unet"])
    p.add_argument("--n-classes", type=int, default=7)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--activation", default="relu6", choices=["relu6", "silu"])
    p.add_argument("--sfpn-path", default=None, help="可选：加载原项目 sfpnr50.pth 骨干权重")
    # 训练
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--loss", default="dice", choices=["dice", "dice_ce", "dice_wce"])
    p.add_argument("--class-weights", type=float, nargs=7, default=None,
                   help="dice_wce 的类别权重（7 个数，对应输出通道 0-6）")
    p.add_argument("--patience", type=int, default=6, help="val mIoU 早停轮数（短训建议 99 关闭早停）")
    p.add_argument("--no-scheduler", dest="use_scheduler", action="store_false", default=True,
                   help="关闭 ReduceLROnPlateau 调度器，lr 全程恒定（短训建议加：保证各组比较的是初始 lr）")
    p.add_argument("--seed", type=int, default=42)
    # 输出
    p.add_argument("--outdir", default="runs")
    p.add_argument("--run-name", default=None, help="本次实验标识（建议：模型_配置编号，如 r50_lr1e-4_adamw_dice）")
    p.add_argument("--resume", default=None,
                   help="断点续训：传入上次保存的 *_last.pth（会恢复模型/优化器/调度器/早停状态，续写同一 CSV；建议连同原 --run-name 一起传入）")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(name, model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=1e-5)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    raise ValueError(name)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total, batches = 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        if not torch.isfinite(loss):
            print("[警告] 跳过无效 loss 批次")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    meter = SegmentationMetrics(model.decoder.out.out_channels
                                if hasattr(model, "decoder") else model.out.out_channels)
    total, batches = 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        if torch.isfinite(loss):
            total += loss.item()
            batches += 1
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        meter.update(pred, masks.cpu().numpy())
    return total / max(batches, 1), meter


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.backbone}_lr{args.lr:g}_{args.optimizer}_{args.loss}"
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{run_name}_log.csv")
    ckpt_best = os.path.join(args.outdir, f"{run_name}_best.pth")
    ckpt_last = os.path.join(args.outdir, f"{run_name}_last.pth")

    print(f"[启动] 设备: {device} | 实验: {run_name}")
    print(f"   配置: backbone={args.backbone} lr={args.lr} opt={args.optimizer} "
          f"loss={args.loss} act={args.activation} epochs={args.epochs} seed={args.seed} "
          f"sched={'on' if args.use_scheduler else 'off'}")

    # 数据
    train_set = RemoteSensingDataset(args.train_img, args.train_mask, args.imgsz)
    val_set = RemoteSensingDataset(args.val_img, args.val_mask, args.imgsz)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 模型 / 损失 / 优化器
    model = build_model(args.backbone, args.n_classes, args.pretrained,
                        args.freeze_backbone, args.activation).to(device)
    if args.sfpn_path:
        load_sfpn_weights(model.backbone, args.sfpn_path)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"   参数量: 总 {total_p:,} | 可训练 {trainable:,}（模型大小对比用）")

    criterion = build_loss(args.loss, args.class_weights)
    optimizer = build_optimizer(args.optimizer, model, args.lr)
    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
                 if args.use_scheduler else None)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_miou, patience_cnt, start_epoch = -1.0, 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_miou = ckpt.get("best_miou", best_miou)
        patience_cnt = ckpt.get("patience_cnt", patience_cnt)
        print(f"[续训] 断点续训: {args.resume} | 从第 {start_epoch + 1} 轮继续 | best val mIoU={best_miou:.4f}")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_loss",
                                    "val_mIoU", "val_PA", "epoch_seconds"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, meter = validate(model, val_loader, criterion, device)
        m = meter.compute()
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(m["mIoU"])

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, f"{cur_lr:.6g}", f"{train_loss:.4f}",
                                    f"{val_loss:.4f}", f"{m['mIoU']:.4f}",
                                    f"{m['PA']:.4f}", f"{elapsed:.0f}"])
        print(f"[数据] Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} | mIoU={m['mIoU']:.4f} PA={m['PA']:.4f} "
              f"| lr={cur_lr:.2g} | {elapsed:.0f}s")

        # ===== 核心修改：按 val mIoU 保存最优模型（原实现按 val loss 保存） =====
        if m["mIoU"] > best_miou:
            best_miou = m["mIoU"]
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "config": vars(args), "epoch": epoch + 1,
                        "val_mIoU": m["mIoU"], "val_PA": m["PA"]}, ckpt_best)
            print(f"[OK] 新的最优模型（val mIoU={best_miou:.4f}）已保存: {ckpt_best}")
            print(meter.summary_str(CLASS_NAMES_7))
        else:
            patience_cnt += 1

        # _last.pth 保存完整续训状态（放在最优判断之后，保证 best/patience 不滞后）
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "scaler_state_dict": scaler.state_dict(),
                    "best_miou": best_miou, "patience_cnt": patience_cnt,
                    "config": vars(args), "epoch": epoch + 1,
                    "val_mIoU": m["mIoU"], "val_PA": m["PA"]}, ckpt_last)

        if m["mIoU"] <= best_miou and patience_cnt >= args.patience:
            print(f"[早停] val mIoU 连续 {args.patience} 轮未提升，早停（第 {epoch + 1} 轮）")
            break

    print(f"\n[完成] 训练结束 | 最优 val mIoU={best_miou:.4f} | 最优权重: {ckpt_best}")
    print(f"   训练曲线 CSV: {csv_path}")


if __name__ == "__main__":
    main()
