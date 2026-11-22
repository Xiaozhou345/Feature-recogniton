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


def _fuse_pyramid(levels):
    f1, f2, f3 = levels
    f2 = f2 + F.interpolate(f3, size=f2.shape[-2:], mode='nearest')
    return f1 + F.interpolate(f2, size=f1.shape[-2:], mode='nearest')


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

    def forward(self, x, *, return_shallow=False, return_pyramid=False):
        x = self.conv1(x)
        x = self.bn1(x)
        shallow = self.relu(x)      # 池化前特征：[B,64,H/2,W/2]
        x = self.maxpool(shallow)

        feat1 = self.layer1(x)      # [B,256,H/4,W/4]
        feat2 = self.layer2(feat1)  # [B,512,H/8,W/8]
        feat3 = self.layer3(feat2)  # [B,1024,H/16,W/16]

        levels = (self.fpn1(feat1), self.fpn2(feat2), self.fpn3(feat3))
        if return_pyramid:
            return levels
        fused = _fuse_pyramid(levels)
        if return_shallow:
            return fused, shallow
        return fused


class ResNet50SFPN(nn.Layer):
    """ResNet50 FPN 骨干 + 移动端 Decoder（对应原 MobileRemoteSensingModel）"""

    def __init__(self, n_classes=7, pretrained=True, freeze_backbone=True,
                 activation="relu6", *, shallow_skip=False, adaptive_fusion=False):
        super().__init__()
        self.backbone = LightweightFPN(pretrained)
        self.decoder = MobileDecoder(n_classes, activation=activation, shallow_skip=shallow_skip)
        self.shallow_skip = shallow_skip
        self.scale_gate = None
        if adaptive_fusion:
            # 权重从 (1, 1, 1) 开始，保留原 FPN 的逐级相加。
            self.scale_gate = nn.Conv2D(
                3 * 128, 3, kernel_size=1, bias_attr=False,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)),
            )

        if freeze_backbone:
            freeze_layer(self.backbone)

    def forward(self, x):
        size = x.shape[-2:]
        if self.scale_gate is not None:
            levels = self.backbone(x, return_pyramid=True)
            pooled = paddle.concat([F.adaptive_avg_pool2d(level, 1) for level in levels], axis=1)
            weights = 3 * F.softmax(self.scale_gate(pooled), axis=1)
            weighted = [level * weights[:, i:i + 1] for i, level in enumerate(levels)]
            out = self.decoder(_fuse_pyramid(weighted))
        elif self.shallow_skip:
            feat, shallow = self.backbone(x, return_shallow=True)
            out = self.decoder(feat, shallow)
        else:
            out = self.decoder(self.backbone(x))
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out                       # [B,7,H,W]
