# -*- coding: utf-8 -*-
"""LeafScan 启动入口。

图形界面：
    python run_leafscan.py
    python run_leafscan.py --weights ..\\PlantDocDetection\\weights\\plant-best.pt --img-size 480

无界面批量（可直接用于评估脚本或服务器端）：
    python run_leafscan.py --cli PlantDocDetection\\data\\images --save-dir out
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LeafScan 叶片病害检测工作台")
    parser.add_argument("--weights", default=None, help="权重文件路径（.pt），默认用上游 weights/plant-best.pt")
    parser.add_argument("--img-size", type=int, default=480, help="推理分辨率，默认 480")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    parser.add_argument("--cli", metavar="IMAGE_DIR", default=None, help="不开界面，批量检测该目录下的图片")
    parser.add_argument("--save-dir", default=None, help="--cli 模式下把画好框的图存到这里")
    return parser.parse_args(argv)


def run_cli(args) -> int:
    from leafscan.detector import Detector, imread_unicode, imwrite_unicode

    detector = Detector(
        weights=Path(args.weights) if args.weights else None,
        img_size=args.img_size,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )
    source = Path(args.cli)
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"目录下没有图片：{source}")
        return 1

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for path in files:
        frame = imread_unicode(path)
        if frame is None:
            print(f"[跳过] {path.name} 读取失败")
            continue
        detections, cost_ms = detector.infer(frame)
        total_boxes += len(detections)
        summary = ", ".join(f"{k}×{v}" for k, v in Detector.summarize(detections).items()) or "无目标"
        print(f"{path.name}: {len(detections)} 个目标 / {cost_ms:.0f} ms / {summary}")
        if save_dir:
            imwrite_unicode(save_dir / path.name, detector.render(frame, detections))

    print(f"共 {len(files)} 张，检出 {total_boxes} 个目标")
    return 0


def run_gui(args) -> int:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from leafscan.workbench import MainWindow

    # 这两个属性必须在 QApplication 实例化之前设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(weights=Path(args.weights) if args.weights else None, img_size=args.img_size)
    window.show()
    return app.exec_()


def _preflight() -> int:
    """启动前检查当前解释器有没有依赖。

    本机同时存在系统 Python 3.11 和 venv38（只装了 torch/numpy 的那一个），
    新开的终端里 `python` 往往落到 3.11，报错是 ModuleNotFoundError: numpy，
    看起来像程序坏了。这里提前拦一下，直接把"该用哪个解释器"打出来。
    """
    import importlib.util

    missing = []
    for mod in ("numpy", "cv2", "torch"):
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    if not missing:
        # 依赖齐了再查上游。本仓库不含 PlantDocDetection/，缺了会在 import detector
        # 的第一行抛 ModuleNotFoundError（报错指向 models.experimental，看不出该放什么），
        # 所以这里提前拦一次，把"缺什么、放哪里"直接打出来。
        upstream_raw = os.environ.get("YOLO_REPO") or str(
            Path(__file__).resolve().parent.parent / "PlantDocDetection"
        )
        upstream = Path(upstream_raw)
        missing_code = [
            p for p in (upstream / "models" / "experimental.py", upstream / "utils" / "general.py")
            if not p.exists()
        ]
        if missing_code:
            print(f"[LeafScan] 找不到上游 YOLOv5 源码：{upstream}")
            for path in missing_code:
                print(f"[LeafScan]   缺少 {path}")
            print("[LeafScan] 本仓库只含应用层，PlantDocDetection/ 需要自备（见 README §2.0）：")
            print("[LeafScan]   1) 放到与 LeafScan/ 同级的 PlantDocDetection/")
            print("[LeafScan]   2) 或 set YOLO_REPO=D:\\path\\to\\PlantDocDetection")
            return 3
        if not (upstream / "weights" / "plant-best.pt").exists():
            print(f"[LeafScan] 注意：默认权重不存在（{upstream / 'weights' / 'plant-best.pt'}），请用 --weights 指定")
        return 0

    venv_py = Path(__file__).resolve().parent.parent / "venv38" / "Scripts" / "python.exe"
    print(f"[LeafScan] 当前解释器缺少依赖：{', '.join(missing)}")
    print(f"[LeafScan] 当前解释器：{sys.executable}")
    if venv_py.exists():
        print(f'[LeafScan] 请改用："{venv_py}" run_leafscan.py')
        print("[LeafScan] 或者直接双击 run.bat / 按 F5 选 \"LeafScan (GUI)\"")
    else:
        print("[LeafScan] 未找到 venv38，请按 README 第 2 节创建虚拟环境并安装 requirements.txt")
    return 2


def main(argv=None) -> int:
    rc = _preflight()
    if rc:
        return rc
    args = parse_args(argv)
    return run_cli(args) if args.cli else run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
