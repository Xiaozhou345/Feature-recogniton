"""数据集（PaddlePaddle 版）——源自 high_model/recognition.py 的 RemoteSensingDataset。
修改点：
1. 输入尺寸参数化（默认仍为 1024）
2. 返回原始标签(0-7)，0=无数据区域；损失与评测内部处理重映射
3. 去除 torchvision transforms，改用 PIL + numpy 等效实现
   （图像 BILINEAR 缩放、/255、mean=std=0.5 归一化到 [-1,1]；掩码 NEAREST 缩放）
"""
import os
import numpy as np
import paddle
from paddle.io import Dataset
from PIL import Image

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def _preprocess_image(img: Image.Image, imgsz: int) -> np.ndarray:
    """PIL 图像 -> float32 (3,imgsz,imgsz)，归一化到 [-1,1]（等效原 torchvision transforms）"""
    if img.size != (imgsz, imgsz):
        img = img.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)


class _ImageTransform:
    """兼容 infer.py 的 build_transform 用法：PIL 图像 -> paddle Tensor (3,imgsz,imgsz)"""

    def __init__(self, imgsz=1024):
        self.imgsz = imgsz

    def __call__(self, img: Image.Image) -> paddle.Tensor:
        return paddle.to_tensor(_preprocess_image(img, self.imgsz), dtype='float32')


def build_transform(imgsz=1024):
    return _ImageTransform(imgsz)


class RemoteSensingDataset(Dataset):
    """遥感图像 + 像素级掩码。标签体系：背景=1,建筑物=2,道路=3,水域=4,荒地=5,森林=6,农田=7（0=无数据，忽略）

    augment=True 时启用训练增强（仅训练集传入）：水平/垂直翻转、90° 旋转、
    亮度对比度抖动；几何变换图像与掩码同步，验证/测试集不受影响。
    """

    def __init__(self, img_dir, mask_dir, imgsz=1024, augment=False):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.imgsz = imgsz
        self.augment = augment
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
        print(f"[数据] 数据集: 图像 {len(self.img_names)} 张, 有效配对 {len(self.valid_pairs)} 对"
              f"{', 增强开启' if augment else ''}")

    def _augment(self, img: Image.Image, mask: Image.Image):
        """图像与掩码同步的轻量增强（依赖 train.py 里的全局随机种子 42 保证可复现）"""
        if np.random.rand() < 0.5:      # 水平翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.5:      # 垂直翻转
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        k = np.random.randint(4)        # 90°×k 旋转（遥感俯视图无方向先验，均有效）
        if k:
            img = img.rotate(90 * k)
            mask = mask.rotate(90 * k)
        if np.random.rand() < 0.5:      # 亮度/对比度抖动（仅图像，掩码不变）
            arr = np.asarray(img, dtype=np.float32)
            alpha = np.random.uniform(0.9, 1.1)              # 对比度
            beta = np.random.uniform(-0.08, 0.08) * 255.0    # 亮度
            img = Image.fromarray(np.clip(arr * alpha + beta, 0, 255).astype(np.uint8))
        return img, mask

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        mask_img = Image.open(os.path.join(self.mask_dir, mask_name))

        if self.augment:
            image, mask_img = self._augment(image, mask_img)

        img_arr = _preprocess_image(image, self.imgsz)

        mask_np = np.asarray(mask_img).astype(np.int64)
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(-1)
        if mask_np.shape != (self.imgsz, self.imgsz):
            m = Image.fromarray(mask_np.astype(np.uint8))
            mask_np = np.asarray(m.resize((self.imgsz, self.imgsz), Image.NEAREST)).astype(np.int64)

        return (paddle.to_tensor(img_arr, dtype='float32'),
                paddle.to_tensor(mask_np, dtype='int64'))
