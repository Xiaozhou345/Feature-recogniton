"""评测指标——基于混淆矩阵，新增模块（原项目缺失，Pre/报告核心材料）。
指标：PA（像素准确率）、mIoU、每类 IoU / Precision / Recall / Dice。
标签 0（无数据区）在统计时忽略，与 DiceLoss 的 valid_mask 逻辑一致。
"""
import numpy as np


class SegmentationMetrics:
    """accumulate 混淆矩阵后统一计算。类别约定：输出通道 0-6 对应标签 1-7。"""

    def __init__(self, n_classes=7):
        self.n = n_classes
        self.conf = np.zeros((n_classes, n_classes), dtype=np.int64)  # conf[i,j]: 真实i 预测j

    def update(self, pred, target_raw):
        """pred: [H,W] 预测通道号 0-6；target_raw: [H,W] 原始标签 0-7（0 忽略）"""
        pred = np.asarray(pred).ravel()
        target = np.asarray(target_raw).ravel()
        valid = target != 0
        pred, target = pred[valid], target[valid] - 1  # 标签1-7 → 0-6
        idx = (target * self.n + pred).astype(np.int64)
        counts = np.bincount(idx, minlength=self.n * self.n)
        self.conf += counts.reshape(self.n, self.n)

    def compute(self):
        conf = self.conf
        tp = np.diag(conf).astype(np.float64)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = tp / (tp + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            dice = 2 * tp / (2 * tp + fp + fn)
        pa = tp.sum() / max(conf.sum(), 1)

        valid_cls = (tp + fp + fn) > 0  # 测试集中未出现的类别不计入 mIoU
        return {
            "PA": float(pa),
            "mIoU": float(np.nanmean(iou[valid_cls])) if valid_cls.any() else 0.0,
            "per_class_IoU": np.where(valid_cls, np.nan_to_num(iou), np.nan).tolist(),
            "per_class_Precision": np.where(valid_cls, np.nan_to_num(precision), np.nan).tolist(),
            "per_class_Recall": np.where(valid_cls, np.nan_to_num(recall), np.nan).tolist(),
            "per_class_Dice": np.where(valid_cls, np.nan_to_num(dice), np.nan).tolist(),
        }

    def reset(self):
        self.conf[:] = 0

    def summary_str(self, class_names=None):
        r = self.compute()
        names = class_names or [f"class_{i}" for i in range(self.n)]
        lines = [f"PA={r['PA']:.4f}  mIoU={r['mIoU']:.4f}", "-" * 46]
        lines.append(f"{'类别':<12}{'IoU':>8}{'P':>8}{'R':>8}{'Dice':>8}")
        for i, name in enumerate(names):
            lines.append(f"{name:<12}{r['per_class_IoU'][i]:>8.4f}"
                         f"{r['per_class_Precision'][i]:>8.4f}"
                         f"{r['per_class_Recall'][i]:>8.4f}"
                         f"{r['per_class_Dice'][i]:>8.4f}")
        return "\n".join(lines)


CLASS_NAMES_7 = ['Background', 'Building', 'Road', 'Water', 'Wasteland', 'Forest', 'Farmland']
