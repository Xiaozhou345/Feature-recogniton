"""数据集——源自 high_model/recognition.py 的 RemoteSensingDataset，修改点：
1. 输入尺寸参数化（默认仍为 1024）
2. 返回原始标签(0-7)，0=无数据区域；损失与评测内部处理重映射
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def build_transform(imgsz=1024):
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class RemoteSensingDataset(Dataset):
    """遥感图像 + 像素级掩码。标签体系：背景=1,建筑物=2,道路=3,水域=4,荒地=5,森林=6,农田=7（0=无数据，忽略）"""

    def __init__(self, img_dir, mask_dir, imgsz=1024):
        self.imgsz = imgsz
        self.img_names = [f for f in os.listdir(img_dir)
                          if os.path.splitext(f)[1].lower() in IMG_EXTS]
        self.valid_pairs = []
        for fname in self.img_names:
            base = os.path.splitext(fname)[0]
            for ext in IMG_EXTS:
                mask_path = os.path.join(mask_dir, base + ext)
                if os.path.exists(mask_path):
                    self.valid_pairs.append((fname, base + ext))
                    break
        print(f"📊 数据集: 图像 {len(self.img_names)} 张, 有效配对 {len(self.valid_pairs)} 对")
        self.transform = build_transform(imgsz)

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        mask_np = np.array(Image.open(os.path.join(self.mask_dir, mask_name))).astype(np.int64)

        image = self.transform(image)
        mask = torch.from_numpy(mask_np).long()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.shape != (self.imgsz, self.imgsz):
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                 size=(self.imgsz, self.imgsz), mode='nearest'
                                 ).squeeze().long()
        return image, mask
