"""模型 3：MobileNetV3-SFPN（经典骨干，轻量化对比模型）——PaddlePaddle 版

结构：MobileNetV3-Large 骨干 + 同款 FPN 融合 + 移动端 Decoder（复用模型 1）。
重要：paddle.vision 的 MobileNetV3 结构与 torchvision 不同（conv/blocks/lastconv，
没有 .features 属性）。已实测逐块输出，与 torch 版等效的取样点为：
  conv + blocks[0:3]  -> 24ch,  stride 4
  blocks[3:6]         -> 40ch,  stride 8
  blocks[6:] + lastconv -> 960ch, stride 32
FPN 通道数与 torch 版完全一致（24/40/960）。
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.models as pm

from .common import MobileDecoder, freeze_layer


class MobileNetV3FPN(nn.Layer):
    """MobileNetV3-Large 骨干 + 同款 FPN 融合"""

    def __init__(self, pretrained=True):
        super().__init__()
        mbv3 = pm.mobilenet_v3_large(pretrained=pretrained)
        # FPN 三层取样点：s4 / s8 / s32，覆盖完整 MobileNetV3-Large（约 3.0M 参数）
        self.layer1 = nn.Sequential(mbv3.conv, mbv3.blocks[0:3])    # 24ch,  stride 4
        self.layer2 = mbv3.blocks[3:6]                              # 40ch,  stride 8
        self.layer3 = nn.Sequential(mbv3.blocks[6:], mbv3.lastconv) # 960ch, stride 32（深层语义）

        self.fpn1 = nn.Conv2D(24, 128, 1)
        self.fpn2 = nn.Conv2D(40, 128, 1)
        self.fpn3 = nn.Conv2D(960, 128, 1)

    def forward(self, x):
        feat1 = self.layer1(x)      # [B,24,H/4,W/4]
        feat2 = self.layer2(feat1)  # [B,40,H/8,W/8]
        feat3 = self.layer3(feat2)  # [B,960,H/32,W/32]

        f3 = self.fpn3(feat3)
        f2 = self.fpn2(feat2) + F.interpolate(f3, size=feat2.shape[-2:], mode='nearest')
        f1 = self.fpn1(feat1) + F.interpolate(f2, size=feat1.shape[-2:], mode='nearest')
        return f1


class MobileNetV3SFPN(nn.Layer):
    """MobileNetV3 FPN 骨干 + 移动端 Decoder"""

    def __init__(self, n_classes=7, pretrained=True, freeze_backbone=True, activation="relu6"):
        super().__init__()
        self.backbone = MobileNetV3FPN(pretrained)
        self.decoder = MobileDecoder(n_classes, activation=activation)

        if freeze_backbone:
            freeze_layer(self.backbone)

    def forward(self, x):
        size = x.shape[-2:]
        feat = self.backbone(x)          # [B,128,H/4,W/4]
        out = self.decoder(feat)         # [B,7,H/2,W/2]（中间上采样一次）
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out                       # [B,7,H,W]
