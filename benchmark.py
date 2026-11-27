"""三模型推理性能实测（成员 F）：参数量 / 模型文件大小 / 单张推理耗时。

同一环境、同一输入尺寸下横向对比三个模型，结果补充进对比表。
用法示例：
  # 不传权重也能测（随机初始化，测结构规模与前向耗时）
  python benchmark.py --imgsz 512 --repeats 50

  # 传入正式训练权重（推荐，文件大小与耗时都按真实部署权重计算）
  python benchmark.py --checkpoints runs/r50_final_best.pth \
      runs/vgg16_final_best.pth runs/mbv3_final_best.pth --repeats 50
"""
import os
import csv
import time
import argparse
import tempfile
import numpy as np
import paddle

from models import build_model

DEFAULT_BACKBONES = ["resnet50", "vgg16_unet", "mobilenet_v3"]


def parse_args():
    p = argparse.ArgumentParser(description="三模型推理性能实测（模型大小 / 参数量 / 耗时）")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="训练权重 .pth 列表；不传则按 DEFAULT_BACKBONES 构建随机权重模型")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--n-classes", type=int, default=7)
    p.add_argument("--warmup", type=int, default=10, help="预热次数（不计入耗时）")
    p.add_argument("--repeats", type=int, default=50, help="计时重复次数，取均值")
    p.add_argument("--save-csv", default="benchmark_results.csv")
    return p.parse_args()


def sync(place):
    """GPU 异步执行，计时前后必须同步，否则耗时严重偏小。"""
    if place == "gpu":
        paddle.device.cuda.synchronize()


def model_file_size_mb(model):
    """把 state_dict 落盘，实测部署权重文件大小（与 checkpoint 保存格式一致）。"""
    with tempfile.NamedTemporaryFile(suffix=".pdparams", delete=False) as f:
        path = f.name
    try:
        paddle.save(model.state_dict(), path)
        return os.path.getsize(path) / 1024 / 1024
    finally:
        os.remove(path)


def build_from_checkpoint(path):
    ckpt = paddle.load(path, return_numpy=False)
    cfg = ckpt.get("config", {})
    backbone = cfg.get("backbone", "resnet50")
    imgsz = int(cfg.get("imgsz", 512))
    model = build_model(backbone, int(cfg.get("n_classes", 7)), pretrained=False,
                        freeze_backbone=False, activation=cfg.get("activation", "relu6"))
    model.set_state_dict(ckpt["model_state_dict"])
    return model, backbone, imgsz, float(ckpt.get("val_mIoU", float("nan")))


def benchmark(model, imgsz, place, warmup, repeats):
    x = paddle.randn([1, 3, imgsz, imgsz], dtype="float32")
    model.eval()
    with paddle.no_grad():
        for _ in range(warmup):
            model(x)
        sync(place)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            sync(place)
            times.append((time.perf_counter() - t0) * 1000.0)
    times = np.asarray(times)
    return times.mean(), times.std()


def main():
    args = parse_args()
    place = "gpu" if (paddle.device.is_compiled_with_cuda()
                      and paddle.device.cuda.device_count() > 0) else "cpu"
    paddle.set_device(place)
    gpu_name = paddle.device.cuda.get_device_name(0) if place == "gpu" else "CPU"
    print(f"[环境] device={place} | {gpu_name} | paddle={paddle.__version__}")

    jobs = []
    if args.checkpoints:
        for path in args.checkpoints:
            model, backbone, imgsz, miou = build_from_checkpoint(path)
            jobs.append((os.path.basename(path), model, imgsz,
                         f"val_mIoU={miou:.4f}" if miou == miou else ""))
    else:
        for backbone in DEFAULT_BACKBONES:
            model = build_model(backbone, args.n_classes, pretrained=False,
                                freeze_backbone=False)
            jobs.append((backbone, model, args.imgsz, "随机权重"))

    rows = []
    for name, model, imgsz, note in jobs:
        n_params = sum(int(p.numel()) for p in model.parameters())
        size_mb = model_file_size_mb(model)
        mean_ms, std_ms = benchmark(model, imgsz, place, args.warmup, args.repeats)
        print(f"[实测] {name:<28} 参数量={n_params / 1e6:8.2f}M | "
              f"文件={size_mb:7.2f}MB | 单张={mean_ms:8.2f}±{std_ms:.2f}ms "
              f"({1000 / mean_ms:6.2f} FPS, {imgsz}×{imgsz}) {note}")
        rows.append([name, f"{n_params / 1e6:.2f}", f"{size_mb:.2f}",
                     f"{mean_ms:.2f}", f"{std_ms:.2f}", f"{1000 / mean_ms:.2f}",
                     imgsz, place, gpu_name, note])

    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["模型", "参数量(M)", "文件大小(MB)", "单张耗时(ms)", "耗时标准差(ms)",
                    "FPS", "输入尺寸", "设备", "GPU型号", "备注"])
        w.writerows(rows)
    print(f"\n[完成] 实测结果已保存: {args.save_csv}")


if __name__ == "__main__":
    main()
