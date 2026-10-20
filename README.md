# LeafScan · 植物叶片病害检测工作台

[![Python](https://img.shields.io/badge/Python-3.7%20~%203.11-3776ab?logo=python&logoColor=white)](#)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5%205.15-41cd52)](#)
[![YOLOv5](https://img.shields.io/badge/Model-YOLOv5%20v5.0-00b0ff)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue)](LICENSE)

基于 YOLOv5 + PlantDoc 数据集的桌面端叶片病害检测应用。推理跑在后台线程里，
支持**单张图片、文件夹批量、视频、摄像头**四种输入，结果可筛选、可导出。
应用层（推理管线 / 界面 / 线程模型 / 评估工具）为本人重写，模型权重与上游 YOLOv5 源码为第三方材料。

> **本仓库只包含应用层代码。** 上游 `PlantDocDetection/`（YOLOv5 源码 + 一个加密界面组件）、
> 权重 `plant-best.pt` 和 PlantDoc 数据集都不在仓库里，`clone` 后需要按
> **[LeafScan/README.md §2.0](LeafScan/README.md#20-前置条件把上游放回去)** 放回上游才能真正运行。
> 原因与许可边界见 [NOTICE.md](LeafScan/NOTICE.md)。
>
> 本仓库以 **GPL-3.0** 发布（全文见 [LICENSE](LICENSE)）：应用层代码可自由使用与修改，但因为运行时
> 链接 GPL-3.0 的 YOLOv5，分发整合作品时须同样以 GPL-3.0 提供源码。上游 YOLOv5 源码与权重
> 均不在本仓库内。

## 实测指标

在 PlantDoc 测试集（239 张图 / 454 个标注框 / 27 个有标注类别）上实跑，CPU、`img_size=480`：

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| mAP@0.5 | **0.673** | VOC 单阈值、all-point 插值、27 类平均 |
| Precision / Recall / F1 | 0.582 / 0.659 / 0.618 | 工作点 `conf >= 0.25`（TP 299 / FP 215 / FN 155）|
| 单帧耗时 | **≈170–210 ms** | CPU、含预处理与 NMS（同一台机器多次实测，随负载浮动）|
| 最优 F1 阈值 | 0.631 @ `conf=0.30` | 阈值扫描，见评估报告 |

逐类 AP、类别混淆、阈值扫描表都在 **[LeafScan/docs/eval_report.md](LeafScan/docs/eval_report.md)**，
由 `tools/evaluate.py` 生成、可一键重跑。**注意这是 VOC 口径的 mAP@0.5，不是 COCO 的 mAP@[.5:.95]，不可直接与论文数字比较。**

## 功能

- 单图检测：检测框 + 中文病害名 + 置信度叠加显示
- 文件夹批量：递归子目录，进度实时回报，界面全程可响应
- 视频 / 摄像头：逐帧异步推理，**丢旧帧**而非排队，延迟恒定不累积
- 类别筛选：只过滤已有结果，**不重新推理**
- 表格与图像联动：选中表格行，对应框高亮
- 结果导出：把框与标签烧进图片（中文标签走 `QPainter`，导出文件用英文类名避免乱码）
- CLI 模式：无显示器环境可批量跑
- 环境自检与端到端冒烟测试：`tools/doctor.py`、`tools/smoke_test.py`

## 技术要点

| 主题 | 做法 |
| --- | --- |
| 坐标一致性 | `letterbox` 预处理与 `scale_coords` 反变换必须成对使用、传同一个 `input_shape`，否则框整体偏移；冒烟测试里做了边界校验 |
| 线程模型 | 推理、模型加载、批量各自独立 `QThread`；单槽位覆盖式喂帧，避免延迟单调增长与内存堆积 |
| 分层 | `workbench → workers → detector` 单向依赖，`detector` 不含任何 Qt 代码，可被脚本或服务直接复用 |
| 可维护性 | 30 个类别名放 `labels_zh.yaml`，不硬编码；重建下拉框用 `blockSignals` 而不是反复 `disconnect/connect` |
| 评估 | 上游 `utils/metrics.py` 已缺失，按 VOC 口径用 numpy 从零实现 AP（贪心匹配 + all-point 插值），并附阈值扫描与类别混淆表 |
| 环境适配 | Windows 中文用户名导致 `tempfile` 非 ASCII 路径让 scipy 在 import 阶段崩溃；在包 `__init__` 里早于任何第三方 import 重定向 `TEMP/TMP`；同时固定解释器避免落到没装 torch 的 Python |

## 目录结构

```
<repo>/
├─ LeafScan/                 应用层源码（本仓库主体）
│  ├─ run_leafscan.py        入口：GUI / CLI
│  ├─ run.bat                双击启动（固定解释器）
│  ├─ leafscan/              detector（纯推理）/ workers（线程）/ workbench（界面）
│  ├─ tools/                 doctor（自检）/ smoke_test（端到端）/ evaluate（mAP）
│  ├─ docs/eval_report.md    实跑评估报告
│  ├─ requirements.txt
│  ├─ README.md              详细说明：环境、实现细节、踩坑、设计问答
│  └─ NOTICE.md              代码来源与第三方许可
├─ LICENSE                   GPL-3.0 全文（仓库整体按此许可）
├─ pyrightconfig.json        Pylance 的 extraPaths / venv 配置
└─ .vscode/                  launch.json：F5 直接跑 GUI / CLI / 冒烟测试 / 自检
```

## 快速开始

```bash
# 1) 按 LeafScan/README.md §2.0 把上游 PlantDocDetection/ 与权重放回同级位置
# 2) 安装依赖（含上游 utils/general.py 需要的 scipy / matplotlib / Pillow / tqdm）
pip install -r LeafScan/requirements.txt

# 3) 自检 → 端到端冒烟测试 → 启动
python LeafScan/tools/doctor.py
python LeafScan/tools/smoke_test.py
python LeafScan/run_leafscan.py                       # 图形界面
python LeafScan/run_leafscan.py --cli <图片目录> --save-dir <输出目录>   # 无界面批量
```

上游可以放在任意位置，用 `YOLO_REPO` 环境变量指过去即可。
详细的参数、目录约定与排错步骤见 **[LeafScan/README.md](LeafScan/README.md)**。

## 我做了什么 / 没做什么

这一节是刻意写的，避免把第三方成果说成自己的：

| 部分 | 来源 |
| --- | --- |
| YOLOv5 v5.0 训练/推理框架 | Ultralytics 开源（GPL-3.0），本仓库只调用其推理期函数 |
| 预训练权重 `plant-best.pt` | PlantDoc 数据集微调产物，**非本仓库训练** |
| PlantDoc 数据集 | Singh et al., *PlantDoc*, CoDS-COMAD 2020 |
| `leafscan/`、`tools/`、`run_leafscan.py` | 本人编写（推理管线、界面、线程模型、评估脚本、冒烟测试） |
| 环境排障与启动固化 | 本人排查修复 |

完整的逐文件说明见 [NOTICE.md](LeafScan/NOTICE.md)。

## 已知限制

- 推理后端只支持 PyTorch；换 ONNX Runtime / TensorRT 可再快 2–4 倍（接口不用改）
- 实时流与批量不并发：共用同一个 `Detector`，批量期间会禁用其它入口（PyTorch 模块多线程前向不安全）
- 摄像头模式无目标跟踪，每帧独立检测
- 评估口径为 VOC 单阈值，未接 COCO 的 mAP@[.5:.95]
- 界面固定中文，未做 i18n；摄像头固定设备 0

## 后续计划

- [ ] `export.py` 导出 ONNX，`detector.py` 增加 `backend="onnx"` 分支
- [ ] 阈值滑条：把扫描结果接成界面可调参数
- [ ] 视频结果导出 mp4
- [x] 评估脚本与逐类 AP / 混淆 / 阈值扫描报告
- [ ] 单元测试：`letterbox` + `scale_coords` 的往返一致性
