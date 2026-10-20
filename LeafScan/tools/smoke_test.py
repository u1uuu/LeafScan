# -*- coding: utf-8 -*-
"""冒烟测试：不用你手动点界面，验证 模型加载 -> 推理 -> 坐标还原 -> 界面构建 全链路。

用法：
    python tools\\smoke_test.py
    python tools\\smoke_test.py --gui      # 额外验证界面能构建（离屏模式，不弹窗）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def build_sample() -> np.ndarray:
    """优先用上游目录里真实的叶片图；找不到就用合成图，保证脚本永远能跑。"""
    from leafscan.detector import UPSTREAM_REPO, imread_unicode

    candidates = [
        p
        for p in UPSTREAM_REPO.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "icons" not in p.parts
    ]
    for path in sorted(candidates):
        image = imread_unicode(path)
        if image is not None and min(image.shape[:2]) >= 200:
            print(f"样例图：{path}")
            return image
    print("未找到可用样例图，改用合成图（检测结果为空属正常）")
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="额外构建一次主窗口（离屏）")
    args = parser.parse_args()

    from leafscan.detector import Detector

    detector = Detector()
    print(f"权重  ：{detector.weights}")
    print(f"设备  ：{detector.device}  half={detector.half}  img_size={detector.img_size}")
    print(f"类别数：{len(detector.names)}")

    frame = build_sample()
    detections, cost_ms = detector.infer(frame)
    print(f"推理  ：{cost_ms:.1f} ms，目标 {len(detections)} 个")
    for det in detections[:5]:
        print(f"  {det.zh_name:<14}{det.confidence:.3f}  {det.box}")

    height, width = frame.shape[:2]
    for det in detections:
        assert 0 <= det.x1 < det.x2 <= width, f"x 坐标越界：{det.box}"
        assert 0 <= det.y1 < det.y2 <= height, f"y 坐标越界：{det.box}"
    print("坐标范围检查：通过")

    rendered = detector.render(frame, detections)
    assert rendered.shape == frame.shape and rendered is not frame, "render 必须返回同尺寸副本"
    print("导出绘制检查：通过")

    if args.gui:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        from leafscan.workbench import MainWindow

        app = QApplication([])
        window = MainWindow()
        window.show()
        window._apply_result(frame, detections)
        window.view.repaint()
        print(f"MainWindow 构建：通过，表格 {window.table.rowCount()} 行")
        print(f"筛选下拉框：{window.combo_filter.count()} 项")
        QTimer.singleShot(1200, app.quit)
        app.exec_()
        window.close()
        print("GUI 冒烟测试：通过")

    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
