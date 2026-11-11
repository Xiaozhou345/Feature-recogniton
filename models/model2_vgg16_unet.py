"""模型 2：VGG16-UNet（经典网络）

结构：VGG16 前五个卷积块做编码器，UNet 风格跳连解码器。
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

from .common import get_activation, _tv_weights


class DoubleConv(nn.Module):
    """UNet 标准双卷积块：Conv3x3-BN-Act x2"""

    def __init__(self, in_ch, out_ch, activation="relu6"):
        super().__init__()
        act = get_activation(activation)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class VGG16UNet(nn.Module):
    """VGG16 前五个卷积块做编码器，UNet 风格跳连解码"""

    def __init__(self, n_classes=7, pretrained=True, freeze_encoder=False, activation="relu6"):
        super().__init__()
        vgg = tv_models.vgg16(weights=_tv_weights(pretrained)).features
        # VGG16 features 索引：0-3 block1 | 4 pool1 | 5-8 block2 | 9 pool2 |
        # 10-15 block3 | 16 pool3 | 17-22 block4 | 23 pool4 | 24-29 block5 | 30 pool5
        self.enc1 = vgg[:4]     # 64ch,  stride 1
        self.enc2 = vgg[4:9]    # 128ch, stride 2
        self.enc3 = vgg[9:16]   # 256ch, stride 4
        self.enc4 = vgg[16:23]  # 512ch, stride 8
        self.enc5 = vgg[23:30]  # 512ch, stride 16

        if freeze_encoder:
            for m in (self.enc1, self.enc2, self.enc3, self.enc4, self.enc5):
                for p in m.parameters():
                    p.requires_grad = False

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec5 = DoubleConv(512 + 512, 512, activation)
        self.dec4 = DoubleConv(512 + 256, 256, activation)
        self.dec3 = DoubleConv(256 + 128, 128, activation)
        self.dec2 = DoubleConv(128 + 64, 64, activation)
        self.dec1 = DoubleConv(64, 64, activation)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                 # [B,64,H,W]
        e2 = self.enc2(e1)                # [B,128,H/2,W/2]
        e3 = self.enc3(e2)                # [B,256,H/4,W/4]
        e4 = self.enc4(e3)                # [B,512,H/8,W/8]
        e5 = self.enc5(e4)                # [B,512,H/16,W/16]

        d5 = self.dec5(torch.cat([self.up(e5), e4], dim=1))
        d4 = self.dec4(torch.cat([self.up(d5), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
        d1 = self.dec1(d2)                # d2 已与输入同分辨率，无需再上采样
        return self.out(d1)               # [B,n_classes,H,W]
