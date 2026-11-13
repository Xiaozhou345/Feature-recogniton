"""数据集统计（成员B）——统计张数与每类像素占比，输出 CSV。

注意：正式锁定数据的统计由 prepare_dataset.py 产出，见 dataset_lock/data_stats_*.csv。
本脚本用于快速核查任意目录，不依赖 torch，可单独运行。

用法示例：
  python data_stats.py --img-dir <images> --mask-dir <masks> --split train

修复点（相对初版）：
1. 去掉 from dataset import IMG_EXTS —— 该导入会连带加载 torch，导致无 GPU 环境无法统计；
2. 越界标签（如 8）不再被静默丢弃，而是单独统计并明确告警，避免占比总数对不上却不报错。
"""
import os
import csv
import argparse
import numpy as np
from PIL import Image

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
CLASS_NAMES_7 = ['Background', 'Building', 'Road', 'Water',
                 'Wasteland', 'Forest', 'Farmland']
LABEL_NAMES = ["NoData(0)"] + CLASS_NAMES_7  # 标签 0-7
VALID_MAX = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--split", default="dataset", help="标记：train / val / test")
    p.add_argument("--save-csv", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)
    illegal = {}
    n_images, n_pairs = 0, 0

    for fname in os.listdir(args.img_dir):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
            continue
        n_images += 1
        base = os.path.splitext(fname)[0]
        mask_path = next((os.path.join(args.mask_dir, base + e) for e in IMG_EXTS
                          if os.path.exists(os.path.join(args.mask_dir, base + e))), None)
        if mask_path is None:
            print(f"警告: 未找到配对掩码: {fname}")
            continue
        n_pairs += 1
        mask = np.array(Image.open(mask_path))
        ids, c = np.unique(mask, return_counts=True)
        for i, v in zip(ids, c):
            i = int(i)
            if 0 <= i <= VALID_MAX:
                counts[i] += int(v)
            else:
                illegal[i] = illegal.get(i, 0) + int(v)

    total = counts.sum()             # 全部像素，含 NoData(0)
    valid_total = counts[1:].sum()   # 有效像素，仅标签 1-7
    print(f"\n===== {args.split} 统计 =====")
    print(f"图像 {n_images} 张 | 有效配对 {n_pairs} 对")
    print(f"全部像素 {total:,} | 有效像素 {valid_total:,} | "
          f"NoData(0) {counts[0]:,} 像素，占全像素 {counts[0] / max(total, 1) * 100:.2f}%")
    print("口径: 下表占比以有效像素为分母，与 dataset_lock/data_stats_*.csv 一致")
    print(f"{'类别':<16}{'像素数':>14}{'有效像素占比':>14}")
    for i in range(1, len(LABEL_NAMES)):
        print(f"{LABEL_NAMES[i]:<16}{counts[i]:>14,}{counts[i] / max(valid_total, 1) * 100:>13.2f}%")

    if illegal:
        n_bad = sum(illegal.values())
        print(f"警告: 发现越界标签 {illegal}，共 {n_bad:,} 像素未计入占比。")
        print("      这些像素会让 losses.py 的交叉熵与 metrics.py 的混淆矩阵越界，")
        print("      请先按 DATASET_PLAN.md 的口径清洗（标签 8 -> 0）再统计。")

    csv_path = args.save_csv or f"data_stats_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label_id", "class", "pixels",
                    "ratio_pct_valid", "ratio_pct_all"])
        w.writerow([args.split, 0, "NoData", int(counts[0]), "",
                    f"{counts[0] / max(total, 1) * 100:.4f}"])
        for i in range(1, len(LABEL_NAMES)):
            w.writerow([args.split, i, LABEL_NAMES[i], int(counts[i]),
                        f"{counts[i] / max(valid_total, 1) * 100:.4f}",
                        f"{counts[i] / max(total, 1) * 100:.4f}"])
        for i, n in sorted(illegal.items()):
            w.writerow([args.split, i, f"ILLEGAL_{i}", int(n), "excluded", "excluded"])
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
