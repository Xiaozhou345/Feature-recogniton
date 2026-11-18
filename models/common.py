"""公共模块：激活函数、轻量化卷积块、移动端 Decoder、权重加载（PaddlePaddle 版）。

模型 1 / 2 / 3 共用的组件集中在这里，各模型文件只定义自己的网络结构。
框架迁移说明：torch.nn -> paddle.nn，requires_grad=False -> stop_gradient=True。
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

ACTIVATIONS = {"relu6": nn.ReLU6}


class SiLU(nn.Layer):
    """paddle 2.6 无 nn.SiLU 层，用 F.silu 函数实现（等效 x * sigmoid(x)）"""

    def forward(self, x):
        return F.silu(x)


ACTIVATIONS["silu"] = SiLU


def get_activation(name: str = "relu6") -> nn.Layer:
    name = (name or "relu6").lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"未知激活函数: {name}，可选 {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


def freeze_layer(layer: nn.Layer):
    """冻结骨干：stop_gradient=True 等价于 torch 的 requires_grad=False"""
    for p in layer.parameters():
        p.stop_gradient = True


class DepthwiseSeparableConv(nn.Layer):
    """深度可分离卷积（移动端核心轻量化模块）"""

    def __init__(self, in_channels, out_channels, stride=1, activation="relu6"):
        super().__init__()
        self.depthwise = nn.Conv2D(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias_attr=False)
        self.pointwise = nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MobileDecoder(nn.Layer):
    """移动端 Decoder（模型 1 / 3 共用，源自原实现）"""

    def __init__(self, n_classes=7, activation="relu6"):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(128, 64, activation=activation)
        self.conv2 = DepthwiseSeparableConv(64, 32, activation=activation)
        self.conv3 = DepthwiseSeparableConv(32, 16, activation=activation)
        self.out = nn.Conv2D(16, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)                                            # [B,64,H/4,W/4]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)                                            # [B,32,H/2,W/2]
        x = self.conv3(x)                                            # [B,16,H,W]（FPN 特征本身为 H/4，需再上采样）
        return self.out(x)


def load_sfpn_weights(model, weights_path, prefix_to_remove="en."):
    """加载权重（按形状匹配）。注意：原项目 sfpnr50.pth 为 torch pickle 格式，
    协议已禁用旧项目权重；本函数仅支持 paddle.save 保存的字典。"""
    weights = paddle.load(weights_path)
    weight_dict = {k.replace(prefix_to_remove, ""): v for k, v in weights.items()}
    loaded, mismatched = 0, []
    for name, param in model.named_parameters():
        if name in weight_dict:
            arr = np.asarray(weight_dict[name])
            if arr.shape == tuple(param.shape):
                param.set_value(arr)
                loaded += 1
            else:
                mismatched.append(name)
    print(f"[OK] SFPN 权重匹配: 成功 {loaded} 个, 形状不匹配 {len(mismatched)} 个")
    return loaded > 0
