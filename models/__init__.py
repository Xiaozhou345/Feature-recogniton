"""模型包：三个模型各占一个文件，统一由 build_model 构建。

- model1_resnet50_sfpn.py  模型 1：ResNet50-SFPN-DW（改进模型）
- model2_vgg16_unet.py     模型 2：VGG16-UNet（经典网络）
- model3_mobilenetv3_sfpn.py 模型 3：MobileNetV3-SFPN（经典骨干）
- common.py                三模型共用的激活函数 / 卷积块 / Decoder / 权重加载

三个模型共用同一套数据 / 损失 / 评测管线，仅骨干网络不同。
"""
from .common import load_sfpn_weights
from .model1_resnet50_sfpn import ResNet50SFPN, LightweightFPN
from .model2_vgg16_unet import VGG16UNet
from .model3_mobilenetv3_sfpn import MobileNetV3SFPN, MobileNetV3FPN

__all__ = [
    "build_model", "load_sfpn_weights",
    "ResNet50SFPN", "VGG16UNet", "MobileNetV3SFPN",
    "LightweightFPN", "MobileNetV3FPN",
]

# 兼容旧名（train.py / eval.py / infer.py 均通过 build_model 构建模型）
SFPNSegModel = ResNet50SFPN


def build_model(backbone="resnet50", n_classes=7, pretrained=True,
                freeze_backbone=True, activation="relu6"):
    """统一入口：backbone ∈ {resnet50, vgg16_unet, mobilenet_v3}"""
    if backbone == "resnet50":
        return ResNet50SFPN(n_classes=n_classes, pretrained=pretrained,
                            freeze_backbone=freeze_backbone, activation=activation)
    if backbone == "mobilenet_v3":
        return MobileNetV3SFPN(n_classes=n_classes, pretrained=pretrained,
                               freeze_backbone=freeze_backbone, activation=activation)
    if backbone == "vgg16_unet":
        return VGG16UNet(n_classes=n_classes, pretrained=pretrained,
                         freeze_encoder=freeze_backbone, activation=activation)
    raise ValueError(f"未知骨干: {backbone}，可选 resnet50 / vgg16_unet / mobilenet_v3")
