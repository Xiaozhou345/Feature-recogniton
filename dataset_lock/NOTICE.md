# 数据集锁定通知（正式版，可直接复制发群）

> 本文是仓库内的正式通知文本，供全组复制发群；`DATASET_PLAN.md` 第 9 节保留同内容的简版模板。成员 B 的个人答辩准备材料不进仓库。

---

【数据集已锁定 v1，请全员切换】

统一使用 `E:\AIshixun\dataset_locked_v1\train|val|test\images_png|masks_png`，划分清单见仓库 `dataset_lock/split_manifest.csv`，**锁定后不再改动**，任何改动需全组同意并重跑全部实验。

规模：train 2519 / val 837 / test 835（60/20/20，主导类分层系统抽样）。图像与掩码已离线重采样为 512×512（图像双线性 / 掩码最近邻）；含标签 8 的 235 张掩码已清洗为 0 忽略。

校验：`dataset_lock/VALIDATION.txt` 通过（落盘 512×512、三分互斥、掩码取值 ⊆ {0..7}、占比最大差 0.43 pp）。

协议：三个入口仍保留 `--imgsz 512` 作双保险（数据本身已是 512）；短训 8 epoch + `--patience 99`；正式训练上限 20 epoch + patience=5；seed=42、bs=1 不变。

类别权重（`dice_wce` 起点）：`0.2869 1.0000 1.7235 1.0425 1.9462 0.5980 0.3803`。

1809 张官方无标签 Test 仅供 F 演示，不进任何指标；旧权重 `sfpnr50.pth` / `soba.pth` 禁用。

数据与方案已提交（commit `17ff650`）；1.69 GB 数据本体在锁定目录，F 上传 AI Studio 时单独拷贝，不要传进 git。
