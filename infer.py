"""单图推理演示（成员F 的答辩演示脚本）——改造自 high_model/recognition_test.py。
修改点：从统一训练框架的 checkpoint 加载（含 config），支持三个 backbone。
用法示例：
  python infer.py --checkpoint runs/resnet50_best.pth --image <图片路径>
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from models import build_model
from dataset import build_transform
from metrics import CLASS_NAMES_7

CLASS_COLORS = np.array([
    [245, 245, 245],  # Background
    [220, 20, 60],    # Building
    [128, 128, 128],  # Road
    [30, 144, 255],   # Water
    [255, 165, 0],    # Wasteland
    [0, 128, 0],      # Forest
    [255, 255, 0],    # Farmland
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/resnet50_best.pth")
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--save", default="inference_result.png")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(args.image).convert("RGB")
    img_tensor = build_transform(args.imgsz)(img).unsqueeze(0).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(cfg.get("backbone", "resnet50"), cfg.get("n_classes", 7),
                        pretrained=False, freeze_backbone=False,
                        activation=cfg.get("activation", "relu6")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ 模型: {args.checkpoint} | backbone={cfg.get('backbone')} | 训练时 val mIoU={ckpt.get('val_mIoU')}")

    pred_mask = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu().numpy()
    pred_color = CLASS_COLORS[pred_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(img); ax1.set_title("Original Image"); ax1.axis("off")
    ax2.imshow(pred_color); ax2.set_title("Segmentation Result"); ax2.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c / 255.0) for c in CLASS_COLORS]
    plt.figlegend(handles, CLASS_NAMES_7, loc="lower center", ncol=7, fontsize=11)
    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    plt.show()

    print("\n📊 分割结果统计:")
    total = pred_mask.size
    for cls_id, count in zip(*np.unique(pred_mask, return_counts=True)):
        print(f"   {CLASS_NAMES_7[cls_id]}: {count:,} 像素 ({count / total * 100:.2f}%)")
    print(f"\n🎉 结果已保存: {args.save}")


if __name__ == "__main__":
    main()
