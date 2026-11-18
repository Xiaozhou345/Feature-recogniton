"""模型 1：ResNet50-SFPN-DW（改进模型）——PaddlePaddle 版

结构：ResNet50 骨干（conv1~layer3）+ 三层 1x1 FPN 融合 + 深度可分离卷积 Decoder。
paddle.vision 的 resnet50 与 torchvision 同构（conv1/bn1/relu/maxpool/layer1-3），
通道数 256/512/1024 一致，已实测验证。
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.models as pm

from .common import MobileDecoder, freeze_layer


class LightweightFPN(nn.Layer):
    """ResNet50 骨干（conv1~layer3）+ 三层 1x1 FPN 融合（原实现）"""

    def __init__(self, pretrained=True):
        super().__init__()
        resnet50 = pm.resnet50(pretrained=pretrained)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1  # 64→256, stride 4
        self.layer2 = resnet50.layer2  # 256→512, stride 8
        self.layer3 = resnet50.layer3  # 512→1024, stride 16

        self.fpn1 = nn.Conv2D(256, 128, 1)
        self.fpn2 = nn.Conv2D(512, 128, 1)
        self.fpn3 = nn.Conv2D(1024, 128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)      # [B,256,H/4,W/4]
        feat2 = self.layer2(feat1)  # [B,512,H/8,W/8]
        feat3 = self.layer3(feat2)  # [B,1024,H/16,W/16]

        f3 = self.fpn3(feat3)
        f2 = self.fpn2(feat2) + F.interpolate(f3, size=feat2.shape[-2:], mode='nearest')
        f1 = self.fpn1(feat1) + F.interpolate(f2, size=feat1.shape[-2:], mode='nearest')
        return f1


class ResNet50SFPN(nn.Layer):
    """ResNet50 FPN 骨干 + 移动端 Decoder（对应原 MobileRemoteSensingModel）"""

    def __init__(self, n_classes=7, pretrained=True, freeze_backbone=True, activation="relu6"):
        super().__init__()
        self.backbone = LightweightFPN(pretrained)
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
