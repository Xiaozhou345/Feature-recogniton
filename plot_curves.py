"""训练曲线绘制：读取 train.py 产出的 *_log.csv，绘制损失曲线与 mIoU/PA 曲线
用法示例：
  # 单个实验
  python plot_curves.py --csv runs/resnet50_lr0.0001_adamw_dice_log.csv
  # 多个实验叠加对比（lr 搜索对比曲线）
  python plot_curves.py --csv runs/r50_c01_log.csv runs/r50_c02_log.csv runs/r50_c03_log.csv --out runs/r50_lr_search.png
  # 自动读取 runs/ 下全部日志
  python plot_curves.py --indir runs --out runs/all_curves.png
"""
import os
import csv
import argparse

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser(description="绘制训练曲线（loss / val mIoU / val PA）")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", nargs="+", help="一个或多个 *_log.csv 路径")
    g.add_argument("--indir", help="目录：自动读取其中全部 *_log.csv")
    p.add_argument("--out", default=None, help="输出 PNG 路径（默认与 CSV 同目录的 curves.png）")
    return p.parse_args()


def collect_csvs(args):
    if args.csv:
        paths = args.csv
    else:
        paths = sorted(os.path.join(args.indir, f)
                       for f in os.listdir(args.indir) if f.endswith("_log.csv"))
        if not paths:
            raise SystemExit(f"[错误] {args.indir} 下没有找到 *_log.csv")
    for pth in paths:
        if not os.path.isfile(pth):
            raise SystemExit(f"[错误] 文件不存在: {pth}")
    return paths


def read_log(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (ValueError, TypeError):
                continue  # 跳过表头残留或坏行
    return rows


def label_of(path):
    return os.path.basename(path).removesuffix("_log.csv").removesuffix(".csv")


def main():
    args = parse_args()
    paths = collect_csvs(args)
    logs = [(label_of(p), read_log(p)) for p in paths]
    single = len(logs) == 1

    out = args.out or os.path.join(os.path.dirname(paths[0]) or ".", "curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if single:
        name, rows = logs[0]
        ep = [r["epoch"] for r in rows]
        fig.suptitle(f"训练曲线：{name}", fontsize=13)
        ax1.plot(ep, [r["train_loss"] for r in rows], label="train_loss")
        ax1.plot(ep, [r["val_loss"] for r in rows], label="val_loss")
        ax1.set_title("Loss")
        ax2.plot(ep, [r["val_mIoU"] for r in rows], label="val_mIoU")
        if "val_PA" in rows[0]:
            ax2.plot(ep, [r["val_PA"] for r in rows], label="val_PA")
        ax2.set_title("验证集指标")
    else:
        for name, rows in logs:
            ep = [r["epoch"] for r in rows]
            ax1.plot(ep, [r["val_loss"] for r in rows], label=name)
            ax2.plot(ep, [r["val_mIoU"] for r in rows], label=name)
        ax1.set_title("val_loss（多个配置对比）")
        ax2.set_title("val mIoU（多个配置对比）")

    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[完成] 曲线已保存: {out}（{len(logs)} 个实验）")


if __name__ == "__main__":
    main()
