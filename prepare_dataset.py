"""成员 B：数据集清洗、分层三分、统计、离线重采样与校验（不改动原始数据）。

解决的问题（见 DATASET_PLAN.md）：
1. 官方 Test 的 1809 张图没有掩码，无法计算独立测试集 mIoU
   -> 从 4191 张带掩码图中重新划分 train/val/test
2. 现有 train/val 类别分布漂移明显 -> 按图像主导类分层 + 系统抽样
3. 掩码存在官方定义外的标签 8 -> 统一映射为 0（与 no-data 一样忽略）
4. 原始掩码含 8 会让 DiceCELoss / SegmentationMetrics 越界崩溃

用法（默认 dry-run，只读、不写任何文件）：
  python prepare_dataset.py
  python prepare_dataset.py --apply --imgsz 512 --out E:\\AIshixun\\dataset_locked_v1

--apply 产物（全部写在新目录，原始 1024 PNG 保持不动）：
  <out>/train|val|test/images_png|masks_png/   供 train.py / eval.py 直接使用
  <out>/split_manifest.csv                     锁定清单，全组不得再改
  <out>/data_stats_<split>.csv                 每个 split 的类别像素占比与样本量
  <out>/class_weights.txt                      dice_wce 的 7 个类别权重（建议起点）
  <out>/VALIDATION.txt                         校验报告

两种落盘模式（本项目推荐 --imgsz 512，见 DATASET_PLAN.md 第 2.1 节）：
  --imgsz 0（默认）：图像/干净掩码用 NTFS 硬链接（零额外磁盘），失败回退复制；
                     含标签 8 的掩码重写为清洗后的 1024 PNG。
  --imgsz 512      ：图像按 BILINEAR、掩码按 NEAREST 离线重采样后落盘，
                     同时完成标签 8 清洗。掩码绝不能用双线性（会造出伪类别）。

统计与类别权重始终按原始 1024 掩码计算；实测缩放前后各类占比偏差 <0.025%。
"""
import os
import csv
import shutil
import argparse
import numpy as np
from PIL import Image

CLASS_NAMES_7 = ["Background", "Building", "Road", "Water",
                 "Wasteland", "Forest", "Farmland"]
SRC_DIRS = [
    ("Train/Train/images_png", "Train/Train/masks_png"),
    ("Val/Val/images_png", "Val/Val/masks_png"),
]
SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"E:\AIshixun\分割DA_high_photo")
    p.add_argument("--out", default=r"E:\AIshixun\dataset_locked_v1")
    p.add_argument("--pattern", default="0,0,0,1,2",
                   help="桶内循环分配模式，0/1/2 对应 train/val/test；默认 60/20/20")
    p.add_argument("--imgsz", type=int, default=0,
                   help="落盘尺寸；0=保持1024并硬链接，512=离线生成512数据集（本项目推荐）")
    p.add_argument("--apply", action="store_true", help="真正生成锁定目录（默认只读预演）")
    return p.parse_args()


def scan(data_root):
    records, seen = [], set()
    for ip, mp in SRC_DIRS:
        img_dir, mask_dir = os.path.join(data_root, ip), os.path.join(data_root, mp)
        imgs, masks = sorted(os.listdir(img_dir)), sorted(os.listdir(mask_dir))
        if set(imgs) != set(masks):
            raise RuntimeError(f"图像与掩码文件名不对齐: {ip}")
        for f in imgs:
            if f in seen:
                raise RuntimeError(f"重复文件名，无法保证三分互斥: {f}")
            seen.add(f)
            b = np.array(Image.open(os.path.join(mask_dir, f)))
            n8 = int((b == 8).sum())
            b[b == 8] = 0                       # 清洗口径：8 -> 0（忽略）
            ids, cnt = np.unique(b, return_counts=True)
            px = np.zeros(7, dtype=np.int64)
            for v, n in zip(ids, cnt):
                if 1 <= v <= 7:
                    px[v - 1] = int(n)
                elif v != 0:
                    raise RuntimeError(f"出现 0-8 之外的标签 {v}: {f}")
            eff = px.sum()
            if eff == 0:
                raise RuntimeError(f"整图无有效标签: {f}")
            records.append({
                "fname": f,
                "img": os.path.join(img_dir, f),
                "mask": os.path.join(mask_dir, f),
                "px": px, "eff": eff, "n8": n8,
                "ratio": px / eff,
                "dom": int(np.argmax(px)),
            })
    return records


def assign(records, pattern):
    """按主导类分 7 桶，桶内按该主类占比排序后循环分配。完全确定性，无需随机种子。"""
    labels = np.full(len(records), -1, dtype=np.int64)
    for k in range(7):
        idx = np.array([i for i, r in enumerate(records) if r["dom"] == k])
        idx = idx[np.argsort([records[i]["ratio"][k] for i in idx], kind="stable")]
        for j, pos in enumerate(idx):
            labels[pos] = pattern[j % len(pattern)]
    if (labels < 0).any():
        raise RuntimeError("存在未分配的样本")
    return labels


def summarize(records, labels):
    print(f"\n总样本 {len(records)} 张")
    rows = []
    for s in range(3):
        sel = [(r, r["px"]) for r, l in zip(records, labels) if l == s]
        px = np.sum([p for _, p in sel], axis=0)
        ratio = px / px.sum() * 100
        imgcnt = np.sum([p > 0 for _, p in sel], axis=0)
        rows.append((SPLIT_NAMES[s], len(sel), ratio, imgcnt, px))
        print(f"  {SPLIT_NAMES[s]:<5} {len(sel):>5} 张 | 各类占比(%) "
              + " ".join(f"{v:5.2f}" for v in ratio))
        print(f"  {'':<5} {'':>5}    | 含该类图像数 "
              + " ".join(f"{v:5d}" for v in imgcnt))
    diff = max(np.abs(rows[i][2] - rows[j][2]).max() for i in range(3) for j in range(3))
    print(f"  三分各类像素占比两两最大差: {diff:.2f} 个百分点")
    n8_files = sum(1 for r in records if r["n8"] > 0)
    n8_px = sum(r["n8"] for r in records)
    print(f"  标签 8 清洗: {n8_files} 张 / {n8_px:,} 像素 -> 0")

    tr = rows[0][2] / 100
    med = float(np.median(tr[tr > 0]))
    w = np.divide(med, tr, out=np.zeros_like(tr), where=tr > 0)
    print("  建议类别权重（中位频率法，dice_wce 起点）: "
          + ", ".join(f"{v:.4f}" for v in w))
    return rows, w, diff


def write_outputs(out, records, labels, rows, weights, diff, imgsz=0):
    for s in SPLIT_NAMES:
        for sub in ("images_png", "masks_png"):
            os.makedirs(os.path.join(out, s, sub), exist_ok=True)

    stat = {"link": 0, "copy": 0, "clean": 0, "resized": 0}
    for r, s in zip(records, labels):
        split = SPLIT_NAMES[s]
        di = os.path.join(out, split, "images_png", r["fname"])
        dm = os.path.join(out, split, "masks_png", r["fname"])

        if imgsz > 0:
            # 离线模式：图像双线性、掩码最近邻；掩码先清洗 8 -> 0
            im = Image.open(r["img"]).convert("RGB")
            im.resize((imgsz, imgsz), Image.Resampling.BILINEAR).save(di)
            b = np.array(Image.open(r["mask"]))
            b[b == 8] = 0
            Image.fromarray(b.astype(np.uint8)).resize(
                (imgsz, imgsz), Image.Resampling.NEAREST).save(dm)
            stat["resized"] += 1
            if r["n8"] > 0:
                stat["clean"] += 1
            continue

        try:
            os.link(r["img"], di)
            stat["link"] += 1
        except OSError:
            shutil.copy2(r["img"], di)
            stat["copy"] += 1
        if r["n8"] > 0:
            b = np.array(Image.open(r["mask"]))
            b[b == 8] = 0
            Image.fromarray(b.astype(np.uint8)).save(dm)
            stat["clean"] += 1
        else:
            try:
                os.link(r["mask"], dm)
                stat["link"] += 1
            except OSError:
                shutil.copy2(r["mask"], dm)
                stat["copy"] += 1

    with open(os.path.join(out, "split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fname", "split", "dominant_class", "n8_pixels"] + CLASS_NAMES_7)
        for r, s in zip(records, labels):
            w.writerow([r["fname"], SPLIT_NAMES[s], CLASS_NAMES_7[r["dom"]], r["n8"]]
                       + [f"{v:.6f}" for v in r["ratio"]])

    with open(os.path.join(out, "class_weights.txt"), "w", encoding="utf-8") as f:
        f.write("# dice_wce 类别权重（顺序对应输出通道 0-6 / 标签 1-7）\n")
        f.write("# 统计口径：原始 1024 掩码（缩放前后占比偏差 <0.025%）\n")
        f.write("# 复制到 --class-weights 参数即可；C/D/E 可在调参中自行调整\n")
        f.write(" ".join(f"{v:.4f}" for v in weights) + "\n")

    for name, n, ratio, imgcnt, px in rows:
        with open(os.path.join(out, f"data_stats_{name}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "label_id", "class", "pixels", "ratio_pct", "images_containing"])
            for i, cn in enumerate(CLASS_NAMES_7):
                w.writerow([name, i + 1, cn, int(px[i]), f"{ratio[i]:.4f}", int(imgcnt[i])])

    # 落盘后复检：取值域、尺寸、配对、互斥
    sets = {}
    problems = []
    for s in SPLIT_NAMES:
        di = os.path.join(out, s, "images_png")
        dm = os.path.join(out, s, "masks_png")
        fi, fm = set(os.listdir(di)), set(os.listdir(dm))
        sets[s] = fi
        if fi != fm:
            problems.append(f"{s}: 图/掩码文件名不对齐")
    for a in range(3):
        for b in range(a + 1, 3):
            if sets[SPLIT_NAMES[a]] & sets[SPLIT_NAMES[b]]:
                problems.append(f"{SPLIT_NAMES[a]} 与 {SPLIT_NAMES[b]} 不互斥")
    expect = (imgsz, imgsz) if imgsz > 0 else (1024, 1024)
    for r, s in zip(records, labels):
        split = SPLIT_NAMES[s]
        pm = os.path.join(out, split, "masks_png", r["fname"])
        arr = np.array(Image.open(pm))
        if arr.min() < 0 or arr.max() > 7:
            problems.append(f"{r['fname']}: 清洗后仍存在越界标签")
        if arr.shape != expect:
            problems.append(f"{r['fname']}: 掩码尺寸 {arr.shape} != {expect}")
        pi = os.path.join(out, split, "images_png", r["fname"])
        if Image.open(pi).size != (expect[1], expect[0]):
            problems.append(f"{r['fname']}: 图像尺寸不符")

    with open(os.path.join(out, "VALIDATION.txt"), "w", encoding="utf-8") as f:
        f.write("数据集锁定校验报告\n")
        f.write(f"划分: " + " / ".join(f"{n}={c}" for n, c, _, _, _ in rows) + "\n")
        f.write(f"落盘尺寸: {imgsz if imgsz > 0 else 1024} x {imgsz if imgsz > 0 else 1024}"
                f"（{'离线重采样' if imgsz > 0 else '原始分辨率硬链接'}）\n")
        f.write(f"三分各类像素占比最大差: {diff:.2f} pp（按原始 1024 掩码统计）\n")
        f.write(f"标签8清洗张数: {stat['clean']}\n")
        f.write(f"硬链接 {stat['link']} / 复制 {stat['copy']} / 离线重采样 {stat['resized']}\n")
        f.write("校验结果: " + ("通过" if not problems else "失败\n" + "\n".join(problems)) + "\n")
    print(f"\n已生成锁定目录: {out}")
    print(f"  硬链接 {stat['link']} / 复制 {stat['copy']} / 离线重采样 {stat['resized']}"
          f" / 清洗掩码 {stat['clean']}")
    print("  校验: " + ("通过" if not problems else "失败 -> " + "; ".join(problems)))
    return problems


def main():
    args = parse_args()
    if args.imgsz not in (0, 512):
        print(f"⚠️ 非标准尺寸 {args.imgsz}；本项目协议只约定 1024 与 512")
    pattern = [int(x) for x in args.pattern.split(",")]
    if any(x not in (0, 1, 2) for x in pattern) or len(pattern) < 3:
        raise ValueError("--pattern 必须是 0/1/2 组成的至少 3 项，如 0,0,0,1,2")
    records = scan(args.data_root)
    labels = assign(records, pattern)
    rows, weights, diff = summarize(records, labels)
    print(f"落盘模式: {'离线生成 ' + str(args.imgsz) + ' 数据集' if args.imgsz > 0 else '原始1024 + 硬链接'}")
    if not args.apply:
        print("\n[dry-run] 未写入任何文件。确认后加 --apply 生成锁定目录。")
        return
    write_outputs(args.out, records, labels, rows, weights, diff, args.imgsz)


if __name__ == "__main__":
    main()
