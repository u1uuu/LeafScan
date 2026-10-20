# -*- coding: utf-8 -*-
"""界面层：用 QPainter 叠加检测框，推理全部走后台线程。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PyQt5.QtCore import QRectF, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .detector import Detection, Detector, imread_unicode, imwrite_unicode, palette
from .workers import BatchWorker, DetectorLoader, InferenceWorker

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


class ImageView(QWidget):
    """等比缩放显示一帧图像，并在其上用 QPainter 叠加检测框。

    不修改原始 ndarray：显示层只负责绘制，数据层不被污染，
    这样"保存结果"导出的图和屏幕上看到的是同一份数据。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._image: Optional[QImage] = None
        self._detections: List[Detection] = []
        self._selected = -1
        self._offset = (0.0, 0.0)
        self._scale = 1.0
        self.hint = "打开图片 / 视频 / 摄像头开始检测"

    # ---------------------------------------------------------------- 数据
    def set_result(self, frame_bgr: np.ndarray, detections: Sequence[Detection]) -> None:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        height, width = rgb.shape[:2]
        # copy() 必须保留：QImage 不持有 numpy 内存，帧被回收后会画花屏
        self._image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self._detections = list(detections)
        self._selected = -1
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = index
        self.update()

    def clear(self) -> None:
        self._image = None
        self._detections = []
        self._selected = -1
        self.update()

    def is_empty(self) -> bool:
        return self._image is None

    # ---------------------------------------------------------------- 绘制
    def _fit_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        self._scale = scale
        self._offset = ((self.width() - w) / 2.0, (self.height() - h) / 2.0)
        return QRectF(self._offset[0], self._offset[1], w, h)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17181c"))
        if self._image is None:
            painter.setPen(QColor("#7a7f8a"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.hint)
            return

        target = self._fit_rect()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(target, self._image)

        font = QFont(self.font())
        font.setPointSize(9)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        for index, det in enumerate(self._detections):
            b, g, r = palette(det.cls_id)
            color = QColor(r, g, b)
            box = QRectF(
                self._offset[0] + det.x1 * self._scale,
                self._offset[1] + det.y1 * self._scale,
                max(1.0, (det.x2 - det.x1) * self._scale),
                max(1.0, (det.y2 - det.y1) * self._scale),
            )
            active = index == self._selected
            painter.setPen(QPen(color, 3 if active else 2))
            if active:
                painter.setBrush(QColor(r, g, b, 40))
            painter.drawRect(box)
            painter.setBrush(Qt.NoBrush)

            text = det.label
            tw = metrics.horizontalAdvance(text) + 8
            th = metrics.height() + 2
            label_rect = QRectF(box.left(), max(0.0, box.top() - th), tw, th)
            painter.fillRect(label_rect, QColor(0, 0, 0, 165))
            painter.setPen(color)
            painter.drawText(label_rect, Qt.AlignCenter, text)


class MainWindow(QMainWindow):
    """主窗口：工具栏 + 图像视图 + 结果面板。"""

    def __init__(self, weights: Optional[Path] = None, img_size: int = 480) -> None:
        super().__init__()
        self.setWindowTitle("LeafScan 叶片病害检测工作台")
        self.resize(1180, 760)

        self._detector: Optional[Detector] = None
        self._worker: Optional[InferenceWorker] = None
        self._batch: Optional[BatchWorker] = None
        self._loader: Optional[DetectorLoader] = None
        self._capture = None
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._on_frame_timeout)
        self._frame_id = 0
        self._frames = 0
        self._last_cost = 0.0
        self._current_frame: Optional[np.ndarray] = None
        self._current_dets: List[Detection] = []
        self._weights = Path(weights) if weights else None
        self._img_size = img_size

        self._build_ui()
        self._load_model(self._weights)

    def _build_ui(self) -> None:
        toolbar = QToolBar("工具栏", self)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.act_image = QAction("打开图片", self)
        self.act_folder = QAction("批量文件夹", self)
        self.act_video = QAction("打开视频", self)
        self.act_camera = QAction("摄像头", self)
        self.act_model = QAction("切换模型", self)
        self.act_save = QAction("保存结果", self)
        self.act_stop = QAction("停止", self)
        for action in (self.act_image, self.act_folder, self.act_video, self.act_camera, self.act_model):
            toolbar.addAction(action)
            toolbar.addSeparator()
        toolbar.addAction(self.act_save)
        toolbar.addAction(self.act_stop)

        self.act_image.triggered.connect(self.open_image)
        self.act_folder.triggered.connect(self.open_folder)
        self.act_video.triggered.connect(self.open_video)
        self.act_camera.triggered.connect(self.open_camera)
        self.act_model.triggered.connect(self.change_model)
        self.act_save.triggered.connect(self.save_result)
        self.act_stop.triggered.connect(self.stop_stream)
        self.act_stop.setEnabled(False)

        self.view = ImageView(self)

        side = QWidget(self)
        side.setFixedWidth(320)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 8, 8, 8)

        side_layout.addWidget(QLabel("类别筛选"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("显示全部类别")
        self.combo_filter.currentIndexChanged.connect(self._on_filter_changed)
        side_layout.addWidget(self.combo_filter)

        side_layout.addWidget(QLabel("检测结果"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["序号", "类别", "置信度"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        side_layout.addWidget(self.table, 1)

        self.lbl_stats = QLabel("暂无结果")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setStyleSheet("color:#9aa3b2;")
        side_layout.addWidget(self.lbl_stats)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self.view)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("正在加载模型…")

    # ---------------------------------------------------------------- 模型
    def _load_model(self, weights: Optional[Path]) -> None:
        self.statusBar().showMessage("正在加载模型…")
        self._loader = DetectorLoader(weights, self._img_size, 0.25, 0.45, self)
        self._loader.loaded.connect(self._on_model_loaded)
        self._loader.failed.connect(self._on_model_failed)
        self._loader.start()

    def _on_model_loaded(self, detector: Detector) -> None:
        self._detector = detector
        if self._worker is None:
            self._worker = InferenceWorker(detector, self)
            self._worker.detected.connect(self._on_detected)
            self._worker.failed.connect(self.statusBar().showMessage)
            self._worker.start()
        else:
            self._worker.set_detector(detector)
        self.setWindowTitle(
            f"LeafScan 叶片病害检测工作台 — {detector.weights.name}（{len(detector.names)} 类）"
        )
        self.statusBar().showMessage("模型就绪", 3000)
        self._set_busy(False)

    def _on_model_failed(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage(message)
        QMessageBox.critical(self, "模型加载失败", message)

    def _need_detector(self) -> bool:
        if self._worker is None or self._detector is None:
            self.statusBar().showMessage("模型尚未就绪，请稍候", 3000)
            return False
        return True

    def change_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch 权重 (*.pt)")
        if not path:
            return
        self.stop_stream()
        self._set_busy(True)
        self._load_model(Path(path))

    # ---------------------------------------------------------------- 输入
    def open_image(self) -> None:
        if not self._need_detector():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
        )
        if not path:
            return
        frame = imread_unicode(Path(path))
        if frame is None:
            QMessageBox.warning(self, "无法读取", f"文件打不开：{path}")
            return
        self.stop_stream()
        self._frames = 1
        self._frame_id += 1
        self._worker.submit(frame, self._frame_id)
        self.statusBar().showMessage(f"已提交：{Path(path).name}")

    def open_folder(self) -> None:
        if not self._need_detector():
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        self.stop_stream()
        self._batch = BatchWorker(self._detector, Path(folder), IMAGE_SUFFIXES, self)
        self._batch.item_done.connect(self._on_batch_item)
        self._batch.failed.connect(self.statusBar().showMessage)
        self._batch.finished_all.connect(self._on_batch_finished)
        self._batch.start()
        self._set_busy(True)
        self.statusBar().showMessage("批量检测中…")

    def open_video(self) -> None:
        if not self._need_detector():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开视频", "", "视频 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)"
        )
        if not path:
            return
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            QMessageBox.warning(self, "无法打开", f"视频打不开：{path}")
            return
        self._start_stream(capture, Path(path).name)

    def open_camera(self) -> None:
        if not self._need_detector():
            return
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            QMessageBox.warning(self, "无法打开", "没有检测到可用摄像头（或被其他程序占用）")
            return
        self._start_stream(capture, "摄像头")

    def _start_stream(self, capture, source: str) -> None:
        self.stop_stream()
        self._capture = capture
        self._frames = 0
        self._timer.start()
        self.act_stop.setEnabled(True)
        self.statusBar().showMessage(f"正在检测：{source}")

    def stop_stream(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        if self._batch is not None and self._batch.isRunning():
            self._batch.abort()
        self.act_stop.setEnabled(False)

    # ---------------------------------------------------------------- 帧循环
    def _on_frame_timeout(self) -> None:
        """定时器只负责"取帧并提交"，绝不做推理。"""
        if self._capture is None or self._worker is None:
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.stop_stream()
            self.statusBar().showMessage("视频结束或取帧失败", 4000)
            return
        self._frame_id += 1
        self._frames += 1
        # 若上一帧还没推理完，worker 内部会用这一帧覆盖上一帧（丢旧帧，不排队）
        self._worker.submit(frame, self._frame_id)

    def _on_detected(self, frame_id: int, frame, detections: List[Detection], cost_ms: float) -> None:
        # 只显示"刚被处理完的那一帧"，保证图像和框永远属于同一帧，不会错位
        self._last_cost = cost_ms
        self._apply_result(frame, detections)
        suffix = f"  累计 {self._frames} 帧" if self._timer.isActive() else ""
        self.statusBar().showMessage(
            f"推理 {cost_ms:.0f} ms | 目标 {len(detections)} 个{suffix}", 4000
        )

    def _on_batch_item(
        self, path: str, frame, detections: List[Detection], cost_ms: float, done: int, total: int
    ) -> None:
        self._last_cost = cost_ms
        self._apply_result(frame, detections)
        self.statusBar().showMessage(f"批量进度 {done}/{total}  {Path(path).name}", 4000)

    def _on_batch_finished(self, processed: int, failed: int) -> None:
        self._set_busy(False)
        self.statusBar().showMessage(f"批量完成：成功 {processed} 张，失败 {failed} 张", 8000)

    # ---------------------------------------------------------------- 展示
    def _apply_result(self, frame, detections: List[Detection]) -> None:
        self._current_frame = frame
        self._current_dets = list(detections)
        self._refresh_filter()
        shown = self._filtered(self._current_dets)
        self.view.set_result(frame, shown)
        self._refresh_table(shown)

    def _filtered(self, detections: List[Detection]) -> List[Detection]:
        cls_id = self.combo_filter.itemData(self.combo_filter.currentIndex())
        if cls_id is None:
            return list(detections)
        return [d for d in detections if d.cls_id == cls_id]

    def _refresh_filter(self) -> None:
        # 用 blockSignals 而不是反复 disconnect/connect：
        # 后者容易漏连或重连多次，导致一个信号触发好几遍槽函数
        blocked = self.combo_filter.blockSignals(True)
        previous = self.combo_filter.itemData(self.combo_filter.currentIndex())
        self.combo_filter.clear()
        self.combo_filter.addItem("显示全部类别", None)
        seen = set()
        for det in self._current_dets:
            if det.cls_id in seen:
                continue
            seen.add(det.cls_id)
            self.combo_filter.addItem(det.zh_name, det.cls_id)
        index = self.combo_filter.findData(previous) if previous is not None else 0
        self.combo_filter.setCurrentIndex(index if index >= 0 else 0)
        self.combo_filter.blockSignals(blocked)

    def _refresh_table(self, detections: List[Detection]) -> None:
        # 重建表格期间屏蔽 itemSelectionChanged，否则每插一行都会触发一次槽
        self.table.blockSignals(True)
        self.table.setRowCount(len(detections))
        for row, det in enumerate(detections):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(det.zh_name))
            self.table.setItem(row, 2, QTableWidgetItem(f"{det.confidence:.2f}"))
        self.table.blockSignals(False)

        summary = Detector.summarize(detections)
        if summary:
            detail = "  ".join(f"{name}×{count}" for name, count in list(summary.items())[:8])
            self.lbl_stats.setText(f"共 {len(detections)} 个目标\n{detail}")
        else:
            self.lbl_stats.setText("未检测到目标")

    def _on_filter_changed(self, _index: int) -> None:
        # 筛选只影响显示，不重新推理
        if self._current_frame is None:
            return
        shown = self._filtered(self._current_dets)
        self.view.set_result(self._current_frame, shown)
        self._refresh_table(shown)

    def _on_row_selected(self) -> None:
        model = self.table.selectionModel()
        rows = model.selectedRows() if model is not None else []
        self.view.set_selected(rows[0].row() if rows else -1)

    # ---------------------------------------------------------------- 其它
    def save_result(self) -> None:
        if self._current_frame is None or self._detector is None:
            self.statusBar().showMessage("没有可保存的结果", 3000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.jpg", "图片 (*.jpg *.png)")
        if not path:
            return
        canvas = self._detector.render(self._current_frame, self._filtered(self._current_dets))
        if imwrite_unicode(Path(path), canvas):
            self.statusBar().showMessage(f"已保存：{path}", 6000)
        else:
            QMessageBox.warning(self, "保存失败", f"写入失败：{path}")

    def _set_busy(self, busy: bool) -> None:
        for action in (self.act_image, self.act_folder, self.act_video, self.act_camera, self.act_model):
            action.setEnabled(not busy)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_stream()
        if self._worker is not None:
            self._worker.stop()
        if self._batch is not None and self._batch.isRunning():
            self._batch.abort()
            self._batch.wait(3000)
        if self._loader is not None and self._loader.isRunning():
            self._loader.wait(3000)
        super().closeEvent(event)
