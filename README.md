# Feature-recogniton — 遥感图像 7 类地物语义分割与多模型对比

人工智能课程专题研究（专题 4 自主选题）训练框架。
代码源自小组原项目 `high_model/recognition.py`，重构为模块化训练框架。

## 项目结构

```
featureRecognition/
├── models/                          # 模型定义（三模型各一个文件）
│   ├── __init__.py                  # 统一入口 build_model(backbone=...)
│   ├── common.py                    # 共用组件：激活函数 / DW 卷积 / Decoder / 权重加载
│   ├── model1_resnet50_sfpn.py      # 模型1：ResNet50 + 轻量化FPN + DW-Decoder（改进模型）
│   ├── model2_vgg16_unet.py         # 模型2：VGG16 编码器 + UNet 跳连解码（经典网络）
│   └── model3_mobilenetv3_sfpn.py   # 模型3：MobileNetV3-Large + 同款FPN（轻量化对比）
├── dataset.py       # 数据集（标签 0=无数据忽略，1-7 对应 7 类）
├── losses.py        # DiceLoss（原实现）/ Dice+CE 组合（支持类别权重）
├── metrics.py       # 混淆矩阵 → PA / mIoU / 每类 IoU / P / R / Dice（新增）
├── train.py         # 统一训练框架（核心修改：按 val mIoU 保存最优模型）
├── eval.py          # 测试集最终评测
├── data_stats.py    # 数据集统计（每类样本量/像素占比，成员B）
├── infer.py         # 单图推理演示（答辩演示用）
├── PROJECT_PLAN.md  # 项目分工与执行规划
├── tuning_log.xlsx  # 调参记录表模板（12 组预填，三工作表）
├── requirements.txt
└── README.md
```

## 关键修改（相对原 recognition.py，答辩素材）

1. **训练策略改进**：最优 checkpoint 由「按 val loss 保存」改为「按 val mIoU 保存」，使模型选择标准与最终评测指标（mIoU）对齐（train.py 中有注释标记）。
2. **骨干参数化**：`--backbone resnet50|mobilenet_v3|vgg16_unet` 一键切换三个模型，三个模型共用同一套数据管线 / 损失 / 评测代码，保证对比公平。
3. **新增评测体系**：原项目没有任何指标计算，现基于混淆矩阵输出 PA/mIoU/每类指标。
4. **新增 Dice+CE 损失**：业界事实标准（nnU-Net 同款思路），弥补纯 Dice 在 batch=1 下小类别梯度噪声大的缺陷；`dice_wce` 支持按类别占比加权。
5. **激活函数可切换**：`--activation relu6|silu`（调参变量）。
6. **早停**：val mIoU 连续 patience 轮（默认 6）不提升即停止，与 25 epoch 上限配合。

## 统一实验协议（全组遵守）

| 项         | 值                                        |
| ---------- | ----------------------------------------- |
| 随机种子   | 42                                        |
| 输入尺寸   | 1024×1024                                 |
| batch size | 1                                         |
| epoch      | 短训 10 / 正式 25（上限）+ patience=6 早停 |
| 选模型指标 | val mIoU                                  |
| 最终成绩   | 独立测试集 mIoU（主）+ PA + 每类 IoU       |

## 快速开始

```bash
pip install -r requirements.txt

# 1. 数据统计（成员B，三个 split 各跑一次）
python data_stats.py --img-dir <images> --mask-dir <masks> --split train

# 2. 调参短训示例（C/D/E 各自的模型，每模型 4 组）
python train.py --backbone resnet50 \
    --train-img <train_images> --train-mask <train_masks> \
    --val-img <val_images> --val-mask <val_masks> \
    --epochs 10 --lr 1e-4 --optimizer adamw --loss dice \
    --run-name r50_base

# 3. 正式训练（最优配置，epochs 上限 25 + 早停）
python train.py --backbone resnet50 ... --epochs 25 --run-name r50_final

# 4. 测试集评测（E）
python eval.py --checkpoint runs/r50_final_best.pth \
    --test-img <test_images> --test-mask <test_masks>

# 5. 单图演示（F）
python infer.py --checkpoint runs/r50_final_best.pth --image test.png
```

## 调参矩阵（共 12 组短训，见 PROJECT_PLAN.md 第四、五节）

baseline（lr=1e-4、AdamW、DiceLoss、ReLU6）为各模型搜索起点；各负责人在 lr、优化器（AdamW/SGD）、损失（Dice / Dice+CE）、激活函数（ReLU6/SiLU）等维度自行组合尝试，每模型 4 组、以 val mIoU 最高者为最终配置：

- 模型 1 ResNet50-SFPN（C，4 组）
- 模型 2 VGG16-UNet（D，4 组）
- 模型 3 MobileNetV3-SFPN（E，4 组）

所有记录汇入 E 的调参记录表（模型 × 配置 × val mIoU / PA / 时长）。

每组结果记录进 Excel 对比总表（模型 × 配置 × val mIoU / PA / 时长）。
