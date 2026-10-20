# -*- coding: utf-8 -*-
"""纯推理层：输入一张 BGR 图，输出结构化检测结果。

设计要点
--------
1. 这一层不 import 任何 Qt，所以可以被命令行、Web 服务、单元测试直接调用。
2. 预处理用 letterbox，后处理用 scale_coords 把框逆映射回原图坐标，
   两者必须同时使用且必须传同一个 input_shape，否则框会整体偏移。
3. 类别中文名来自 labels_zh.yaml，和模型 names 做对齐，
   不在代码里硬编码 30 个字符串。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_THIS = Path(__file__).resolve()

# YOLOv5 源码根目录（models/、utils/），默认取同级的 PlantDocDetection
UPSTREAM_REPO = Path(
    os.environ.get("YOLO_REPO", str(_THIS.parents[2] / "PlantDocDetection"))
).resolve()

DEFAULT_WEIGHTS = UPSTREAM_REPO / "weights" / "plant-best.pt"
DEFAULT_LABELS = _THIS.parent / "labels_zh.yaml"

if str(UPSTREAM_REPO) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_REPO))

import torch  # noqa: E402

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


@dataclass(frozen=True)
class Detection:
    """单个检测框。坐标是原图像素坐标。"""

    cls_id: int
    name: str
    zh_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def label(self) -> str:
        return f"{self.zh_name} {self.confidence:.2f}"

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def load_zh_labels(path: Path = DEFAULT_LABELS) -> Dict[str, str]:
    """读取“英文类名 -> 中文名”映射；文件缺失时返回空表（退化为英文名）。"""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """cv2.imread 不支持中文路径，用 imdecode 兜底。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """同上，写文件也要走 encode。"""
    ext = os.path.splitext(str(path))[1]
    if not ext:
        return False
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def palette(cls_id: int) -> Tuple[int, int, int]:
    """按类别生成稳定颜色，避免每次刷新都换色。返回 BGR。"""
    import colorsys

    hue = (cls_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class Detector:
    """YOLOv5 检测器封装。

    参数
    ----
    weights     权重路径
    img_size    推理分辨率，必须是 stride 的整数倍（内部用 check_img_size 修正）
    conf_thres  置信度阈值，低于它的候选框先被丢弃
    iou_thres   NMS 的 IoU 阈值，同一目标的重复框在这里被合并
    classes     只保留指定类别 id（None 表示全部）
    """

    def __init__(
        self,
        weights: Optional[Path] = None,
        img_size: int = 480,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        agnostic_nms: bool = False,
        device: str = "",
        half: Optional[bool] = None,
    ) -> None:
        self.weights = Path(weights) if weights else DEFAULT_WEIGHTS
        if not self.weights.exists():
            raise FileNotFoundError(f"权重文件不存在：{self.weights}")

        self.device = select_device(device)
        # half 只对 GPU 有意义；CPU 上强行 half 会报 "not implemented for 'Half'"
        self.half = bool(half) if half is not None else self.device.type != "cpu"

        self.model = attempt_load(str(self.weights), map_location=self.device)
        self.model.eval()
        if self.half:
            self.model.half()

        raw_names = self.model.names
        self.names: Dict[int, str] = (
            dict(raw_names) if isinstance(raw_names, dict) else dict(enumerate(raw_names))
        )
        self.zh_map = load_zh_labels()
        self.img_size = check_img_size(img_size, s=int(self.model.stride.max()))
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.classes = list(classes) if classes is not None else None
        self.agnostic_nms = bool(agnostic_nms)

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], float]:
        """对单帧做检测，返回 (结果列表, 耗时毫秒)。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return [], 0.0

        started = time.time()
        letterboxed, _, _ = letterbox(frame_bgr, new_shape=self.img_size)
        chw = letterboxed[:, :, ::-1].transpose(2, 0, 1)
        chw = np.ascontiguousarray(chw)

        tensor = torch.from_numpy(chw).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor = tensor / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        input_shape = tensor.shape[2:]  # (h, w)，scale_coords 必须用这个

        with torch.no_grad():
            pred = self.model(tensor, augment=False)[0]
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=self.classes,
                agnostic=self.agnostic_nms,
            )

        results: List[Detection] = []
        for det in pred:
            if det is None or not len(det):
                continue
            # 关键：把 letterbox 后的坐标还原到原始帧尺寸
            det[:, :4] = scale_coords(input_shape, det[:, :4], frame_bgr.shape).round()
            for x1, y1, x2, y2, conf, cls in det.tolist():
                cls_id = int(cls)
                name = self.names.get(cls_id, str(cls_id))
                results.append(
                    Detection(
                        cls_id=cls_id,
                        name=name,
                        zh_name=self.zh_map.get(name, name),
                        confidence=float(conf),
                        x1=max(0, int(x1)),
                        y1=max(0, int(y1)),
                        x2=min(frame_bgr.shape[1] - 1, int(x2)),
                        y2=min(frame_bgr.shape[0] - 1, int(y2)),
                    )
                )
        results.sort(key=lambda d: d.confidence, reverse=True)
        return results, (time.time() - started) * 1000.0

    def render(self, frame_bgr: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        """把框烧进图像副本，用于导出结果文件。

        注意：cv2.putText 不支持中文（会变成一堆 ???），所以导出时只画英文类名；
        界面上的中文标签由 QPainter 绘制。
        """
        canvas = frame_bgr.copy()
        for det in detections:
            color = palette(det.cls_id)
            cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            cv2.putText(
                canvas,
                f"{det.name} {det.confidence:.2f}",
                (det.x1 + 2, max(14, det.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def summarize(detections: Sequence[Detection]) -> Dict[str, int]:
        """按中文类名统计数量，供表格与柱状图使用。"""
        counter: Dict[str, int] = {}
        for det in detections:
            counter[det.zh_name] = counter.get(det.zh_name, 0) + 1
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
