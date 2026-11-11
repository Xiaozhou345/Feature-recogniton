"""公共模块：激活函数、轻量化卷积块、移动端 Decoder、权重加载。

模型 1 / 2 / 3 共用的组件集中在这里，各模型文件只定义自己的网络结构。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"relu6": nn.ReLU6, "silu": nn.SiLU}


def get_activation(name: str = "relu6") -> nn.Module:
    name = (name or "relu6").lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"未知激活函数: {name}，可选 {list(ACTIVATIONS)}")
    return ACTIVATIONS[name](inplace=True) if name == "relu6" else ACTIVATIONS[name]()


def _tv_weights(pretrained: bool):
    return "IMAGENET1K_V1" if pretrained else None


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（移动端核心轻量化模块）"""

    def __init__(self, in_channels, out_channels, stride=1, activation="relu6"):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MobileDecoder(nn.Module):
    """移动端 Decoder（模型 1 / 3 共用，源自原实现）"""

    def __init__(self, n_classes=7, activation="relu6"):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(128, 64, activation=activation)
        self.conv2 = DepthwiseSeparableConv(64, 32, activation=activation)
        self.conv3 = DepthwiseSeparableConv(32, 16, activation=activation)
        self.out = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)                                            # [B,64,H/4,W/4]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)                                            # [B,32,H/2,W/2]
        x = self.conv3(x)                                            # [B,16,H,W]（FPN 特征本身为 H/4，需再上采样）
        return self.out(x)


def load_sfpn_weights(model, weights_path, prefix_to_remove="en."):
    """可选：加载原项目 sfpnr50.pth 权重（键名前缀剥离后按形状匹配）"""
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    weight_dict = {k.replace(prefix_to_remove, ""): v for k, v in weights.items()}
    loaded, mismatched = 0, []
    for name, param in model.named_parameters():
        if name in weight_dict and weight_dict[name].shape == param.shape:
            param.data.copy_(weight_dict[name].data)
            loaded += 1
        elif name in weight_dict:
            mismatched.append(name)
    print(f"✅ SFPN 权重匹配: 成功 {loaded} 个, 形状不匹配 {len(mismatched)} 个")
    return loaded > 0
