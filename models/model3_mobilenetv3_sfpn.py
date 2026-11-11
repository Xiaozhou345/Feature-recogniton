"""模型 3：MobileNetV3-SFPN（经典骨干，轻量化对比模型）

结构：MobileNetV3-Large 骨干（前三个尺度层）+ 同款 FPN 融合 + 移动端 Decoder（复用模型 1）。
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .common import MobileDecoder, _tv_weights


class MobileNetV3FPN(nn.Module):
    """MobileNetV3-Large 骨干（前三个尺度层）+ 同款 FPN 融合"""

    def __init__(self, pretrained=True):
        super().__init__()
        m = tv_models.mobilenet_v3_large(weights=_tv_weights(pretrained))
        self.layer1 = m.features[:3]   # 24ch,  stride 4
        self.layer2 = m.features[3:6]  # 40ch,  stride 8
        self.layer3 = m.features[6:10] # 80ch,  stride 16

        self.fpn1 = nn.Conv2d(24, 128, 1)
        self.fpn2 = nn.Conv2d(40, 128, 1)
        self.fpn3 = nn.Conv2d(80, 128, 1)

    def forward(self, x):
        feat1 = self.layer1(x)      # [B,24,H/4,W/4]
        feat2 = self.layer2(feat1)  # [B,40,H/8,W/8]
        feat3 = self.layer3(feat2)  # [B,80,H/16,W/16]

        f3 = self.fpn3(feat3)
        f2 = self.fpn2(feat2) + F.interpolate(f3, scale_factor=2, mode='nearest')
        f1 = self.fpn1(feat1) + F.interpolate(f2, scale_factor=2, mode='nearest')
        return f1


class MobileNetV3SFPN(nn.Module):
    """MobileNetV3 FPN 骨干 + 移动端 Decoder"""

    def __init__(self, n_classes=7, pretrained=True, freeze_backbone=True, activation="relu6"):
        super().__init__()
        self.backbone = MobileNetV3FPN(pretrained)
        self.decoder = MobileDecoder(n_classes, activation=activation)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        size = x.shape[-2:]
        feat = self.backbone(x)          # [B,128,H/4,W/4]
        out = self.decoder(feat)         # [B,7,H/2,W/2]（中间上采样一次）
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out                       # [B,7,H,W]
