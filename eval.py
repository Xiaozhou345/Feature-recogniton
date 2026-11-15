"""测试集评测——在独立测试集上输出最终成绩（PA / mIoU / 每类 IoU / P / R / Dice）。
用法示例：
  python eval.py --checkpoint runs/resnet50_best.pth \
      --test-img <test_images_dir> --test-mask <test_masks_dir>
"""
import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader

from models import build_model
from dataset import RemoteSensingDataset
from losses import build_loss
from metrics import SegmentationMetrics, CLASS_NAMES_7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-img", required=True)
    p.add_argument("--test-mask", required=True)
    p.add_argument("--imgsz", type=int, default=None, help="不传则自动读取 checkpoint 内训练时的 imgsz")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-csv", default=None, help="结果保存路径（默认与 checkpoint 同名 _eval.csv）")
    p.add_argument("--tta", action="store_true", default=False,
                   help="测试时增强：原图/水平翻转/旋转90/旋转270 四次预测取平均（推理约 4 倍耗时）")
    return p.parse_args()


@torch.no_grad()
def predict_logits(model, imgs, tta):
    """单次预测或 4 变换 TTA 融合（结果均变换回原方向后取平均）"""
    if not tta:
        return model(imgs)
    outs = [model(imgs)]
    outs.append(torch.flip(model(torch.flip(imgs, dims=[3])), dims=[3]))
    outs.append(torch.rot90(model(torch.rot90(imgs, k=1, dims=[2, 3])), k=-1, dims=[2, 3]))
    outs.append(torch.rot90(model(torch.rot90(imgs, k=3, dims=[2, 3])), k=1, dims=[2, 3]))
    return torch.stack(outs).mean(dim=0)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    backbone = cfg.get("backbone", "resnet50")
    imgsz = args.imgsz or cfg.get("imgsz", 512)
    model = build_model(backbone, cfg.get("n_classes", 7),
                        pretrained=False, freeze_backbone=False,
                        activation=cfg.get("activation", "relu6")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[OK] 已加载 {args.checkpoint} | backbone={backbone} | imgsz={imgsz} | TTA={'开' if args.tta else '关'} "
          f"| 训练时最优 val mIoU={ckpt.get('val_mIoU', float('nan')):.4f}")

    test_set = RemoteSensingDataset(args.test_img, args.test_mask, imgsz)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    meter = SegmentationMetrics(cfg.get("n_classes", 7))
    for imgs, masks in loader:
        imgs = imgs.to(device)
        pred = torch.argmax(predict_logits(model, imgs, args.tta), dim=1).cpu().numpy()
        meter.update(pred, masks.numpy())

    print("\n===== 测试集最终成绩 =====")
    print(meter.summary_str(CLASS_NAMES_7))
    result = meter.compute()

    csv_path = args.save_csv or os.path.splitext(args.checkpoint)[0] + "_eval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["PA", f"{result['PA']:.4f}"])
        w.writerow(["mIoU", f"{result['mIoU']:.4f}"])
        for i, name in enumerate(CLASS_NAMES_7):
            w.writerow([f"IoU_{name}", f"{result['per_class_IoU'][i]:.4f}"])
            w.writerow([f"P_{name}", f"{result['per_class_Precision'][i]:.4f}"])
            w.writerow([f"R_{name}", f"{result['per_class_Recall'][i]:.4f}"])
    print(f"\n[OK] 结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
