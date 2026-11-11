"""数据集统计（成员B）——统计每张数与每类像素占比，输出 CSV。
用法示例：
  python data_stats.py --img-dir <images_dir> --mask-dir <masks_dir> --split train
"""
import os
import csv
import argparse
import numpy as np
from PIL import Image

from dataset import IMG_EXTS
from metrics import CLASS_NAMES_7

LABEL_NAMES = ["NoData(0)"] + CLASS_NAMES_7  # 标签 0-7


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
    n_images, n_pairs = 0, 0

    for fname in os.listdir(args.img_dir):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
            continue
        n_images += 1
        base = os.path.splitext(fname)[0]
        mask_path = next((os.path.join(args.mask_dir, base + e) for e in IMG_EXTS
                          if os.path.exists(os.path.join(args.mask_dir, base + e))), None)
        if mask_path is None:
            print(f"⚠️ 未找到配对掩码: {fname}")
            continue
        n_pairs += 1
        mask = np.array(Image.open(mask_path))
        ids, c = np.unique(mask, return_counts=True)
        for i, v in zip(ids, c):
            if 0 <= i < len(LABEL_NAMES):
                counts[i] += int(v)

    total = counts.sum()
    print(f"\n===== {args.split} 统计 =====")
    print(f"图像 {n_images} 张 | 有效配对 {n_pairs} 对 | 总像素 {total:,}")
    print(f"{'标签':<16}{'像素数':>14}{'占比':>10}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:<16}{counts[i]:>14,}{counts[i] / max(total, 1) * 100:>9.2f}%")

    csv_path = args.save_csv or f"data_stats_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "pixels", "ratio"])
        for i, name in enumerate(LABEL_NAMES):
            w.writerow([args.split, name, int(counts[i]), f"{counts[i] / max(total, 1) * 100:.4f}"])
    print(f"\n📄 已保存: {csv_path}")


if __name__ == "__main__":
    main()
