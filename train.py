"""统一训练框架（PaddlePaddle 版，核心改动：原按 val loss 保存最优模型 → 改为按 val mIoU 保存）
对应《项目分工与执行规划》第四节的统一实验协议：
- 固定项：seed=42、梯度裁剪、早停；ReduceOnPlateau 调度器可用 --no-scheduler 关闭（短训建议关闭）
- 训练策略改进（答辩素材）：验证集以 mIoU 选最优 checkpoint，与最终评测指标对齐
框架迁移说明：torch -> paddle；AMP 用 paddle.amp；梯度裁剪改由优化器 gradient_clip 实现。
"""
import os
import csv
import time
import random
import argparse
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader

from models import build_model, load_sfpn_weights
from dataset import RemoteSensingDataset
from losses import build_loss
from metrics import SegmentationMetrics, CLASS_NAMES_7


def parse_args():
    p = argparse.ArgumentParser(description="遥感图像 7 类语义分割统一训练框架（PaddlePaddle）")
    # 数据
    p.add_argument("--train-img", required=True)
    p.add_argument("--train-mask", required=True)
    p.add_argument("--val-img", required=True)
    p.add_argument("--val-mask", required=True)
    p.add_argument("--imgsz", type=int, default=512, help="输入尺寸（锁定数据集为 512）")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    # 模型
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "resnet50_shallow", "mobilenet_v3", "vgg16_unet"],
                   help="resnet50_shallow 为模型 1 的浅层跳连变体")
    p.add_argument("--n-classes", type=int, default=7)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--unfreeze-last-stage", action="store_true", default=False,
                   help="统一协议：整体冻结骨干后仅解冻最后一个 stage（ResNet50=layer3、VGG16=enc5、MobileNetV3=layer3），需配合 --augment 使用")
    p.add_argument("--augment", action="store_true", default=False,
                   help="训练集数据增强（翻转/旋转/色彩抖动）；解冻微调时必须开启，验证集不受影响")
    p.add_argument("--activation", default="relu6", choices=["relu6", "silu"])
    p.add_argument("--sfpn-path", default=None, help="可选：加载 paddle.save 格式的骨干权重（torch 旧权重协议已禁用）")
    # 训练
    p.add_argument("--epochs", type=int, default=20, help="v2 协议：正式训练上限 30；短训传 8")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="解冻末段时骨干参数组的学习率倍率（默认 0.1，即骨干 lr = --lr × 0.1；不解冻时无效果）")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--loss", default="dice", choices=["dice", "dice_ce", "dice_wce"])
    p.add_argument("--class-weights", type=float, nargs=7, default=None,
                   help="dice_wce 的类别权重（7 个数，对应输出通道 0-6）")
    p.add_argument("--patience", type=int, default=5, help="val mIoU 早停轮数（短训建议 99 关闭早停；v2 协议正式训练传 8）")
    p.add_argument("--no-scheduler", dest="use_scheduler", action="store_false", default=True,
                   help="关闭 ReduceOnPlateau 调度器，lr 全程恒定（短训建议加：保证各组比较的是初始 lr）")
    p.add_argument("--seed", type=int, default=42)
    # 输出
    p.add_argument("--outdir", default="runs")
    p.add_argument("--run-name", default=None, help="本次实验标识（建议：模型_配置编号，如 r50_c01）")
    p.add_argument("--resume", default=None,
                   help="断点续训：传入上次保存的 *_last.pth（会恢复模型/优化器/调度器/早停状态，续写同一 CSV）")
    return p.parse_args()


def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_optimizer(name, model, lr, use_scheduler, backbone_params=None,
                    backbone_lr_mult=0.1):
    """梯度裁剪通过优化器 gradient_clip 实现（等价 torch 的 clip_grad_norm_）。

    backbone_params 非空时启用双学习率参数组：骨干组 lr = 基础 lr × backbone_lr_mult
    （预训练权重微调用小步长），解码器组保持基础 lr 不变。
    """
    trainable = [p for p in model.parameters() if not p.stop_gradient]
    if backbone_params:
        backbone_ids = {id(p) for p in backbone_params}
        bb_params = [p for p in trainable if id(p) in backbone_ids]
        other_params = [p for p in trainable if id(p) not in backbone_ids]
        params = [
            {"params": bb_params, "params_attr": {"learning_rate": backbone_lr_mult}},
            {"params": other_params},
        ]
        n_bb = sum(p.numel() for p in bb_params)
        print(f"[优化器] 双学习率: 骨干组 {n_bb:,} 参数 × {backbone_lr_mult} | "
              f"解码器组 {sum(p.numel() for p in other_params):,} 参数 × 1.0")
    else:
        params = trainable
    learning_rate = (paddle.optimizer.lr.ReduceOnPlateau(learning_rate=lr, mode='max',
                                                         patience=3, factor=0.5, verbose=False)
                     if use_scheduler else lr)
    clip = nn.ClipGradByGlobalNorm(clip_norm=1.0)
    if name == "adamw":
        return paddle.optimizer.AdamW(learning_rate=learning_rate, parameters=params,
                                      weight_decay=1e-5, grad_clip=clip)
    if name == "sgd":
        return paddle.optimizer.SGD(learning_rate=learning_rate, parameters=params,
                                    momentum=0.9, weight_decay=1e-4, grad_clip=clip)
    raise ValueError(name)


def train_one_epoch(model, loader, criterion, optimizer, scaler, use_amp):
    model.train()
    total, batches = 0.0, 0
    for imgs, masks in loader:
        optimizer.clear_grad()
        with paddle.amp.auto_cast(enable=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        if not bool(paddle.all(paddle.isfinite(loss))):
            print("[警告] 跳过无效 loss 批次")
            continue
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        scaler.update()
        total += float(loss)
        batches += 1
    return total / max(batches, 1)


@paddle.no_grad()
def validate(model, loader, criterion):
    model.eval()
    n_cls = int((model.decoder.out if hasattr(model, "decoder") else model.out).weight.shape[0])
    meter = SegmentationMetrics(n_cls)
    total, batches = 0.0, 0
    for imgs, masks in loader:
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        if bool(paddle.all(paddle.isfinite(loss))):
            total += float(loss)
            batches += 1
        pred = paddle.argmax(outputs, axis=1).cpu().numpy()
        meter.update(pred, masks.cpu().numpy())
    return total / max(batches, 1), meter


def main():
    args = parse_args()
    set_seed(args.seed)
    place = 'gpu' if (paddle.device.is_compiled_with_cuda()
                      and paddle.device.cuda.device_count() > 0) else 'cpu'
    paddle.set_device(place)
    run_name = args.run_name or f"{args.backbone}_lr{args.lr:g}_{args.optimizer}_{args.loss}"
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{run_name}_log.csv")
    ckpt_best = os.path.join(args.outdir, f"{run_name}_best.pth")
    ckpt_last = os.path.join(args.outdir, f"{run_name}_last.pth")

    print(f"[启动] 设备: {place} | 实验: {run_name}")
    print(f"   配置: backbone={args.backbone} lr={args.lr} opt={args.optimizer} "
          f"loss={args.loss} act={args.activation} epochs={args.epochs} seed={args.seed} "
          f"sched={'on' if args.use_scheduler else 'off'} "
          f"unfreeze_last_stage={args.unfreeze_last_stage} augment={args.augment}")

    # 数据
    train_set = RemoteSensingDataset(args.train_img, args.train_mask, args.imgsz,
                                     augment=args.augment)
    val_set = RemoteSensingDataset(args.val_img, args.val_mask, args.imgsz)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, return_list=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, return_list=True)

    # 模型 / 损失 / 优化器
    model = build_model(args.backbone, args.n_classes, args.pretrained,
                        args.freeze_backbone, args.activation,
                        unfreeze_last_stage=args.unfreeze_last_stage)
    if args.sfpn_path:
        load_sfpn_weights(model.backbone, args.sfpn_path)

    # 解冻末段时收集骨干可训练参数（用于双学习率分组）
    backbone_params = None
    if args.unfreeze_last_stage and args.freeze_backbone:
        bb = getattr(model, "backbone", None)
        stage = getattr(bb, "layer3", None) if bb is not None else None
        if stage is None:
            stage = getattr(model, "enc5", None)   # vgg16_unet 无 .backbone 属性
        backbone_params = [p for p in stage.parameters() if not p.stop_gradient]

    trainable = int(sum(p.numel() for p in model.parameters() if not p.stop_gradient))
    total_p = int(sum(p.numel() for p in model.parameters()))
    print(f"   参数量: 总 {total_p:,} | 可训练 {trainable:,}（模型大小对比用）")
    if backbone_params:
        n_bb = sum(p.numel() for p in backbone_params)
        print(f"   解冻末段: 骨干 {n_bb:,} 参数 (lr={args.lr * args.backbone_lr_mult:g})"
              f" | 解码器 {trainable - n_bb:,} 参数 (lr={args.lr:g})")

    criterion = build_loss(args.loss, args.class_weights)
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.use_scheduler,
                                backbone_params=backbone_params,
                                backbone_lr_mult=args.backbone_lr_mult)
    use_amp = (place == 'gpu')
    scaler = paddle.amp.GradScaler(enable=use_amp)

    best_miou, patience_cnt, start_epoch = -1.0, 0, 0
    if args.resume:
        ckpt = paddle.load(args.resume, return_numpy=False)
        model.set_state_dict(ckpt["model_state_dict"])
        optimizer.set_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_miou = float(ckpt.get("best_miou", best_miou))
        patience_cnt = int(ckpt.get("patience_cnt", patience_cnt))
        print(f"[续训] 断点续训: {args.resume} | 从第 {start_epoch + 1} 轮继续 | best val mIoU={best_miou:.4f}")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_loss",
                                    "val_mIoU", "val_PA", "epoch_seconds"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_amp)
        val_loss, meter = validate(model, val_loader, criterion)
        m = meter.compute()
        elapsed = time.time() - t0
        cur_lr = optimizer.get_lr()
        if args.use_scheduler:
            optimizer._learning_rate.step(m["mIoU"])

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, f"{float(cur_lr):.6g}", f"{train_loss:.4f}",
                                    f"{val_loss:.4f}", f"{m['mIoU']:.4f}",
                                    f"{m['PA']:.4f}", f"{elapsed:.0f}"])
        print(f"[数据] Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} | mIoU={m['mIoU']:.4f} PA={m['PA']:.4f} "
              f"| lr={float(cur_lr):.2g} | {elapsed:.0f}s")

        # ===== 核心修改：按 val mIoU 保存最优模型（原实现按 val loss 保存） =====
        if m["mIoU"] > best_miou:
            best_miou = m["mIoU"]
            patience_cnt = 0
            paddle.save({"model_state_dict": model.state_dict(),
                         "config": vars(args), "epoch": epoch + 1,
                         "val_mIoU": m["mIoU"], "val_PA": m["PA"]}, ckpt_best)
            print(f"[OK] 新的最优模型（val mIoU={best_miou:.4f}）已保存: {ckpt_best}")
            print(meter.summary_str(CLASS_NAMES_7))
        else:
            patience_cnt += 1

        # _last.pth 保存完整续训状态（放在最优判断之后，保证 best/patience 不滞后）
        paddle.save({"model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
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
