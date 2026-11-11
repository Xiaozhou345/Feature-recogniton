"""损失函数——源自 high_model/recognition.py 的 DiceLoss，新增 Dice+CE 组合。
修改点：DiceCELoss 支持类别权重（配合 data_stats.py 的类别占比做不均衡加权）。
注意：分割任务的"准确率"(mIoU/PA)不可微，只能作为评测指标，不能当损失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """多类 Dice 损失（忽略标签0的无数据区域），原实现"""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):
        if outputs.shape[2:] != masks.shape[1:]:
            outputs = F.interpolate(outputs, size=masks.shape[1:],
                                    mode='bilinear', align_corners=False)
        valid_mask = (masks != 0).float()
        if valid_mask.sum() < 1:
            return torch.tensor(0.5, device=outputs.device, requires_grad=True)

        outputs = F.softmax(outputs, dim=1)
        total_loss, class_count = 0.0, 0
        for c in range(outputs.shape[1]):
            pred = outputs[:, c] * valid_mask
            target = (masks == (c + 1)).float() * valid_mask
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() + self.smooth
            if union > self.smooth:
                total_loss += 1 - (2 * intersection + self.smooth) / (union + self.smooth)
                class_count += 1
        return total_loss / class_count if class_count > 0 else \
            torch.tensor(0.5, device=outputs.device, requires_grad=True)


class DiceCELoss(nn.Module):
    """Dice + 交叉熵组合（业界事实标准，如 nnU-Net）。
    CE 提供稳定的逐像素梯度，弥补纯 Dice 在小类别/batch=1 下的梯度噪声。
    """
    IGNORE_INDEX = 255  # 标签0(无数据)映射到此并忽略

    def __init__(self, ce_weight=None, smooth=1e-5):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.ce_weight = None if ce_weight is None else torch.as_tensor(ce_weight, dtype=torch.float32)

    def forward(self, outputs, masks):
        if outputs.shape[2:] != masks.shape[1:]:
            outputs = F.interpolate(outputs, size=masks.shape[1:],
                                    mode='bilinear', align_corners=False)
        # CE 部分：标签 1-7 重映射为 0-6，标签 0(无数据)映射为 IGNORE_INDEX 并忽略
        target_ce = masks.clone()
        valid = masks != 0
        target_ce[~valid] = self.IGNORE_INDEX
        target_ce[valid] = target_ce[valid] - 1
        if valid.sum() == 0:
            return torch.tensor(0.5, device=outputs.device, requires_grad=True)
        ce = F.cross_entropy(outputs, target_ce,
                             weight=self.ce_weight.to(outputs.device) if self.ce_weight is not None else None,
                             ignore_index=self.IGNORE_INDEX)
        # Dice 部分
        dice = self.dice(outputs, masks)
        return dice + ce


def build_loss(name="dice", class_weights=None):
    """name: dice | dice_ce | dice_wce（dice_wce = 带类别权重的 Dice+CE）"""
    name = name.lower()
    if name == "dice":
        return DiceLoss()
    if name == "dice_ce":
        return DiceCELoss()
    if name == "dice_wce":
        assert class_weights is not None, "dice_wce 需要提供类别权重（7 个数，对应输出通道 0-6）"
        return DiceCELoss(ce_weight=class_weights)
    raise ValueError(f"未知损失函数: {name}")
