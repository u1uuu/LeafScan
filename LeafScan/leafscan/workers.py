# -*- coding: utf-8 -*-
"""后台线程：推理与模型加载。

为什么不用 QTimer + processEvents
---------------------------------
原做法是在主线程里同步跑推理，再靠 QApplication.processEvents() 强行刷界面。
这会造成两个问题：
    1. 推理期间界面假死，窗口标题栏出现"未响应"；
    2. processEvents 会把按钮点击、定时器、绘制事件重入到推理函数里，
       用户连续点"打开视频"就可能出现两个定时器同时喂帧（原代码里
       self.pass_flag 就是用来堵这个洞的，而且它在 __init__ 里被注释掉了）。

这里的做法：推理放到 QThread，主线程只负责提交帧和接收结果；
待处理帧只保留最新一帧（丢旧帧而不是排队），这样即使推理慢于取帧速度，
也不会积压成越来越长的延迟。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .detector import Detector


class InferenceWorker(QThread):
    """单帧推理线程。同一时刻只处理一帧，多提交的帧会被新的覆盖。"""

    detected = pyqtSignal(int, object, object, float)  # frame_id, frame, [Detection], 耗时ms
    failed = pyqtSignal(str)

    def __init__(self, detector: Detector, parent=None) -> None:
        super().__init__(parent)
        self._detector = detector
        self._frame: Optional[np.ndarray] = None
        self._frame_id = -1
        self._cond = threading.Condition()
        self._stopping = False

    def set_detector(self, detector: Detector) -> None:
        self._detector = detector

    def submit(self, frame: np.ndarray, frame_id: int) -> None:
        with self._cond:
            self._frame = frame
            self._frame_id = frame_id
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify()
        self.wait(3000)

    def run(self) -> None:  # noqa: D102
        while True:
            with self._cond:
                while self._frame is None and not self._stopping:
                    self._cond.wait(0.1)
                if self._stopping:
                    return
                frame = self._frame
                frame_id = self._frame_id
                self._frame = None
            try:
                detections, cost_ms = self._detector.infer(frame)
            except Exception as exc:  # 单帧失败不应该杀掉线程
                self.failed.emit(f"推理失败：{exc}")
                continue
            self.detected.emit(frame_id, frame, detections, cost_ms)


class DetectorLoader(QThread):
    """模型加载线程。换模型时界面不会卡住。"""

    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, weights, img_size: int, conf: float, iou: float, parent=None) -> None:
        super().__init__(parent)
        self._weights = weights
        self._img_size = img_size
        self._conf = conf
        self._iou = iou

    def run(self) -> None:  # noqa: D102
        try:
            detector = Detector(
                weights=self._weights,
                img_size=self._img_size,
                conf_thres=self._conf,
                iou_thres=self._iou,
            )
        except Exception as exc:
            self.failed.emit(f"模型加载失败：{exc}")
            return
        self.loaded.emit(detector)


class BatchWorker(QThread):
    """文件夹批量推理线程。

    注意：和 InferenceWorker 共用同一个 Detector 对象是有意为之（省一次显存），
    所以两者不会同时运行——界面在批量期间禁用其它入口。
    """

    item_done = pyqtSignal(str, object, object, float, int, int)  # path, frame, dets, ms, done, total
    finished_all = pyqtSignal(int, int)  # 成功数, 失败数
    failed = pyqtSignal(str)

    def __init__(self, detector: Detector, folder: Path, suffixes, parent=None) -> None:
        super().__init__(parent)
        self._detector = detector
        self._folder = Path(folder)
        self._suffixes = {s.lower() for s in suffixes}
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:  # noqa: D102
        from .detector import imread_unicode

        files = sorted(
            p for p in self._folder.rglob("*") if p.is_file() and p.suffix.lower() in self._suffixes
        )
        total = len(files)
        if total == 0:
            self.finished_all.emit(0, 0)
            return

        done = 0
        failed = 0
        for path in files:
            if self._abort:
                break
            frame = imread_unicode(path)
            if frame is None:
                failed += 1
                continue
            try:
                detections, cost_ms = self._detector.infer(frame)
            except Exception as exc:
                failed += 1
                self.failed.emit(f"{path.name}: {exc}")
                continue
            done += 1
            self.item_done.emit(str(path), frame, detections, cost_ms, done, total)
        self.finished_all.emit(done, failed)
