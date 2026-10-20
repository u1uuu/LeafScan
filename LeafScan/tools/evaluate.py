# -*- coding: utf-8 -*-
"""在 PlantDoc 测试集上算 mAP@0.5、逐类 AP、以及类别混淆对。

为什么自己写而不用现成的：上游 `PlantDocDetection/utils/` 里没有 `metrics.py`
（YOLOv5 原版的 `ap_per_class` 就在那个文件里，这里被删掉了）。
所以下面用 numpy 实现了 VOC 风格的 all-point 插值 AP：

    1. 同一类别的所有检测按置信度从高到低排序；
    2. 逐个与同图同类的 GT 做贪心匹配，IoU >= 0.5 且该 GT 未被占用算 TP，否则 FP；
    3. 累加得到 PR 曲线，取包络后按 recall 变化点积分。

注意这是**单 IoU 阈值（0.5）**的 VOC 口径，不是 COCO 的 0.5:0.95 平均，
所以数值不能直接和论文里的 COCO mAP 对比。

用法：
    python tools\\evaluate.py
    python tools\\evaluate.py --limit 50          # 先小样本验证流程
    python tools\\evaluate.py --conf 0.001 --op-conf 0.25
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def upstream_root() -> Path:
    """上游目录：`YOLO_REPO` 优先，否则取同级的 PlantDocDetection（与 detector.py 规则一致）。

    不在这里 import detector：那会连带 import torch，`--help` 也要等好几秒。
    """
    return Path(os.environ.get("YOLO_REPO") or (ROOT.parent / "PlantDocDetection")).resolve()


IOU_MATCH = 0.5


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """成对 IoU，返回 (len(a), len(b))。空输入返回空矩阵。"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def voc_ap(is_tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """单个类别的 AP（all-point 插值）。"""
    if n_gt == 0:
        return float("nan")
    if len(is_tp) == 0:
        return 0.0

    order = np.argsort(-conf)
    is_tp = is_tp[order]

    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(~is_tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # 取右侧最大值做成单调不增的包络
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def load_gt(label_path: Path, width: int, height: int) -> List[Tuple[int, np.ndarray]]:
    """读 YOLO 标注（cls cx cy w h，归一化）→ [(cls, xyxy 像素)]。"""
    if not label_path.exists():
        return []

    out: List[Tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            continue
        out.append(
            (
                cls_id,
                np.array(
                    [
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        (cx + bw / 2) * width,
                        (cy + bh / 2) * height,
                    ],
                    dtype=np.float32,
                ),
            )
        )
    return out


def evaluate(args: argparse.Namespace) -> int:
    from leafscan.detector import Detector, imread_unicode

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"在 {images_dir} 下没有找到图片")
        return 2

    weights = Path(args.weights) if args.weights else None
    detector = Detector(
        weights=weights, img_size=args.img_size, conf_thres=args.conf, iou_thres=args.nms_iou
    )
    names = detector.names  # {cls_id: 英文名}
    print(
        f"评估 {len(image_paths)} 张 | conf={args.conf} nms_iou={args.nms_iou} "
        f"img_size={detector.img_size} device={detector.device}"
    )

    per_cls: Dict[int, List[Tuple[bool, float]]] = defaultdict(list)
    n_gt_cls: Counter = Counter()
    confusions: Counter = Counter()
    total_fp = 0
    op_fp = 0
    processed = 0
    elapsed_ms = 0.0
    gt_matched_conf: List[float] = []

    for index, image_path in enumerate(image_paths, 1):
        frame = imread_unicode(image_path)
        if frame is None:
            continue

        height, width = frame.shape[:2]
        gt_boxes: Dict[int, List[np.ndarray]] = defaultdict(list)
        for cls_id, box in load_gt(labels_dir / (image_path.stem + ".txt"), width, height):
            gt_boxes[cls_id].append(box)
            n_gt_cls[cls_id] += 1
        gt_stack = {k: np.stack(v) for k, v in gt_boxes.items()}

        detections, cost_ms = detector.infer(frame)
        processed += 1
        elapsed_ms += cost_ms

        used = {k: np.zeros(len(v), dtype=bool) for k, v in gt_stack.items()}
        for det in sorted(detections, key=lambda d: -d.confidence):
            box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)[None, :]
            matched = False

            same_cls = gt_stack.get(det.cls_id)
            if same_cls is not None:
                ious = iou_matrix(box, same_cls)[0]
                for j in np.argsort(-ious):
                    if ious[j] < IOU_MATCH:
                        break
                    if not used[det.cls_id][j]:
                        used[det.cls_id][j] = True
                        matched = True
                        gt_matched_conf.append(det.confidence)
                        break

            per_cls[det.cls_id].append((matched, det.confidence))
            if matched:
                continue

            total_fp += 1
            if det.confidence >= args.op_conf:
                op_fp += 1

            # 这个 FP 是不是把别的类别检成了当前类别
            best_cls, best_iou = None, IOU_MATCH
            for other_cls, boxes in gt_stack.items():
                if other_cls == det.cls_id:
                    continue
                ious = iou_matrix(box, boxes)[0]
                j = int(np.argmax(ious))
                if ious[j] >= best_iou:
                    best_iou, best_cls = float(ious[j]), other_cls
            if best_cls is not None and det.confidence >= args.op_conf:
                confusions[(det.cls_id, best_cls)] += 1

        if index % 25 == 0 or index == len(image_paths):
            print(f"  {index}/{len(image_paths)} ...")

    if not processed:
        print("一张图都没读进来，检查路径是否包含中文/文件是否存在")
        return 3

    n_gt_total = sum(n_gt_cls.values())
    op_tp = sum(1 for conf in gt_matched_conf if conf >= args.op_conf)
    op_fn = max(n_gt_total - op_tp, 0)
    precision = op_tp / max(op_tp + op_fp, 1)
    recall = op_tp / max(op_tp + op_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # 全局阈值扫描：同一批检测在不同 conf 下的 P/R/F1，用来支撑"阈值怎么定"
    tp_list: List[bool] = []
    conf_list: List[float] = []
    for records in per_cls.values():
        for is_correct, conf in records:
            tp_list.append(is_correct)
            conf_list.append(conf)
    tp_arr = np.array(tp_list, dtype=bool)
    conf_arr = np.array(conf_list, dtype=np.float32)

    sweep: List[Tuple[float, float, float, float, int, int, int]] = []
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        selected = conf_arr >= th
        tp_th = int(tp_arr[selected].sum())
        fp_th = int((~tp_arr[selected]).sum())
        fn_th = max(n_gt_total - tp_th, 0)
        p_th = tp_th / max(tp_th + fp_th, 1)
        r_th = tp_th / max(tp_th + fn_th, 1)
        f1_th = 2 * p_th * r_th / max(p_th + r_th, 1e-9)
        sweep.append((th, p_th, r_th, f1_th, tp_th, fp_th, fn_th))

    rows = []
    for cls_id, records in per_cls.items():
        is_tp = np.array([tp for tp, _ in records], dtype=bool)
        confs = np.array([c for _, c in records], dtype=np.float32)
        ap = voc_ap(is_tp, confs, n_gt_cls.get(cls_id, 0))
        if not np.isnan(ap):
            rows.append((cls_id, ap, n_gt_cls.get(cls_id, 0)))

    # 有 GT 但一个都没检出来的类别，AP 记为 0，不能从 mAP 里漏掉
    for cls_id, count in n_gt_cls.items():
        if count and cls_id not in per_cls:
            rows.append((cls_id, 0.0, count))

    rows.sort(key=lambda r: -r[1])
    mean_ap = float(np.mean([r[1] for r in rows])) if rows else 0.0

    report = build_report(
        args=args,
        detector=detector,
        image_count=processed,
        latency_ms=elapsed_ms / processed,
        rows=rows,
        mean_ap=mean_ap,
        n_gt_total=n_gt_total,
        op=(precision, recall, f1, op_tp, op_fp, op_fn),
        sweep=sweep,
        total_fp=total_fp,
        confusions=confusions,
        names=names,
        zh_map=detector.zh_map,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print()
    print(f"mAP@0.5 = {mean_ap:.4f}   ({len(rows)} 个类别，{n_gt_total} 个 GT 框)")
    print(
        f"工作点 conf>={args.op_conf}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
        f"(TP={op_tp} FP={op_fp} FN={op_fn})"
    )
    print(f"平均单帧耗时 {elapsed_ms / processed:.1f} ms（{processed} 张）")
    print(f"报告已写入 {out_path}")
    return 0


def build_report(
    *,
    args: argparse.Namespace,
    detector,
    image_count: int,
    latency_ms: float,
    rows: Sequence[Tuple[int, float, int]],
    mean_ap: float,
    n_gt_total: int,
    op: Tuple[float, float, float, int, int, int],
    sweep: Sequence[Tuple[float, float, float, float, int, int, int]],
    total_fp: int,
    confusions: Counter,
    names,
    zh_map: Dict[str, str],
) -> str:
    import torch

    precision, recall, f1, op_tp, op_fp, op_fn = op

    def zh(cls_id: int) -> str:
        english = names.get(cls_id, str(cls_id))
        return zh_map.get(english, english)

    out: List[str] = []
    out.append("# PlantDoc 测试集评估报告")
    out.append("")
    out.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    # 报告里不留绝对路径：能被仓库根包含时写成相对路径（PlantDocDetection/weights/...），
    # 上游放在别处（YOLO_REPO）时只写文件名，避免把本机目录结构带进版本库。
    weights_path = Path(detector.weights)
    try:
        weights_display = weights_path.resolve().relative_to(ROOT.parent).as_posix()
    except (ValueError, OSError):
        weights_display = weights_path.name
    out.append(f"- 权重：`{weights_display}`")
    out.append(f"- 环境：torch {torch.__version__} / device `{detector.device}` / img_size {detector.img_size}")
    out.append(f"- 测试集：{image_count} 张图，{n_gt_total} 个标注框，{len(rows)} 个有标注的类别")
    out.append(f"- 推理参数：conf_thres={args.conf}，NMS IoU={args.nms_iou}")
    out.append(f"- 平均单帧耗时：**{latency_ms:.1f} ms**（CPU）")
    out.append("")
    out.append("## 1. 总体指标")
    out.append("")
    out.append("| 指标 | 数值 | 说明 |")
    out.append("| --- | --- | --- |")
    out.append(f"| **mAP@0.5** | **{mean_ap:.4f}** | 对 {len(rows)} 个类别取平均，单 IoU 阈值 0.5 |")
    out.append(f"| Precision | {precision:.3f} | 工作点 conf >= {args.op_conf} |")
    out.append(f"| Recall | {recall:.3f} | 同上 |")
    out.append(f"| F1 | {f1:.3f} | 同上 |")
    out.append(
        f"| TP / FP / FN | {op_tp} / {op_fp} / {op_fn} | 工作点口径；全置信度下 FP 共 {total_fp} |"
    )
    out.append("")
    out.append("## 2. 逐类 AP@0.5")
    out.append("")
    out.append("| # | 类别 | 英文类名 | GT 数 | AP@0.5 |")
    out.append("| --- | --- | --- | --- | --- |")
    for rank, (cls_id, ap, count) in enumerate(rows, 1):
        english = names.get(cls_id, str(cls_id))
        out.append(f"| {rank} | {zh(cls_id)} | {english} | {count} | {ap:.4f} |")
    out.append("")
    out.append("## 3. 类别混淆（预测错到别的类）")
    out.append("")
    if confusions:
        out.append("| 模型预测为 | 实际是 | 次数 |")
        out.append("| --- | --- | --- |")
        for (pred_cls, true_cls), count in confusions.most_common(10):
            out.append(f"| {zh(pred_cls)} | {zh(true_cls)} | {count} |")
        out.append("")
        out.append("> 判定方式：某个 FP 框与另一类别的 GT 的 IoU >= 0.5，即视为\"类别认错\"，"
                   "而不是\"框歪了\"。次数多的那几对，通常是叶片外观高度相似的病害。")
    else:
        out.append("没有出现\"框对但类别错\"的情况。")
    out.append("")
    out.append("## 4. 置信度阈值扫描（同一个模型，只改 conf）")
    out.append("")
    out.append("| conf | Precision | Recall | F1 | TP | FP | FN |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    best_th = max(sweep, key=lambda row: row[3])[0] if sweep else float("nan")
    for th, p, r, f, tp_n, fp_n, fn_n in sweep:
        mark = " **← F1 最高**" if th == best_th else ""
        out.append(f"| {th:.2f} | {p:.3f} | {r:.3f} | {f:.3f}{mark} | {tp_n} | {fp_n} | {fn_n} |")
    out.append("")
    out.append(
        f"> 界面默认 `conf=0.25`，偏召回：宁可多报也不漏检。这批数据上 F1 最高的是 "
        f"`conf={best_th:.2f}`。阈值该往哪边调，取决于用错方向的代价——"
    )
    out.append(
        "> 病害预警场景里误报的代价是人工复核（便宜），漏检的代价是一整片作物（昂贵），所以要保召回；"
        "若是自动喷药，误报的代价反而更高。这类取舍能讲清楚，比报一个 mAP 数字更有说服力。"
    )
    out.append("")
    out.append("## 5. 口径说明（避免误读）")
    out.append("")
    out.append("- 这是 **VOC 风格的单阈值 AP**（IoU=0.5，all-point 插值），")
    out.append("  不是 COCO 的 mAP@[.5:.95]，两者数值不能直接对比。")
    out.append("- 匹配是**贪心**的：同一图上同类别的检测按置信度从高到低，")
    out.append("  与尚未被占用的 GT 做匹配，IoU 达标即算 TP，一个 GT 只能被匹配一次。")
    out.append("- AP 用全部置信度的检测计算（`conf_thres` 设得很低以保留完整 PR 曲线），")
    out.append("  而 P/R/F1 是在工作点（`conf >= op_conf`）上统计的，两个口径分开看才对。")
    out.append("- 有 GT 但整类一个框都没检出的类别，AP 记 0 并计入分母，不会从 mAP 里消失。")
    out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="PlantDoc 测试集评估")
    parser.add_argument("--images", default=str(upstream_root() / "PlantDoc" / "images" / "test"), help="测试图片目录")
    parser.add_argument("--labels", default=str(upstream_root() / "PlantDoc" / "labels" / "test"), help="YOLO 标注目录")
    parser.add_argument("--weights", default=None, help="权重路径，默认上游 plant-best.pt")
    parser.add_argument("--img-size", type=int, default=480, dest="img_size")
    parser.add_argument("--conf", type=float, default=0.001, help="保留检测的置信度下限")
    parser.add_argument("--nms-iou", type=float, default=0.45, dest="nms_iou")
    parser.add_argument("--op-conf", type=float, default=0.25, dest="op_conf", help="统计 P/R/F1 的工作点")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全部")
    parser.add_argument("--out", default=str(ROOT / "docs" / "eval_report.md"))
    args = parser.parse_args()

    started = time.time()
    code = evaluate(args)
    print(f"总耗时 {time.time() - started:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
