# 代码来源与第三方许可

本仓库是在一个既有项目结构上**重建应用层**得到的。为避免含混，先把"哪一行代码从哪来"列清楚：

| 路径 | 来源 | 说明 |
| --- | --- | --- |
| `leafscan/detector.py` | 本仓库作者编写 | 模型加载、`letterbox` 预处理、前向推理、NMS、坐标反变换（`scale_coords`）、中文类名映射 |
| `leafscan/workbench.py` | 本仓库作者编写 | `QGraphicsView` 视图（缩放/平移/框叠加）、结果表格、筛选与导出 |
| `leafscan/workers.py` | 本仓库作者编写 | `QThread` 推理 worker、批量队列、丢帧策略 |
| `leafscan/__init__.py` | 本仓库作者编写 | 含 Windows 非 ASCII 临时目录的处理（见 README §4.6） |
| `run_leafscan.py` | 本仓库作者编写 | 程序入口与命令行分支 |
| `tools/evaluate.py` | 本仓库作者从零编写 | VOC 口径 AP（all-point 插值 + IoU 0.5 贪心匹配）、逐类 AP、类别混淆、阈值扫描 |
| `tools/smoke_test.py` | 本仓库作者从零编写 | 端到端冒烟测试：模型加载 → 推理 → 坐标边界校验 → 界面构建 |
| `docs/` | 本仓库作者编写 / 实跑生成 | 评估报告与工程文档 |

> 重建的原因：原仓库的界面封装依赖下面第 4 节所述的**第三方加密组件**，
> 该组件与当前 Python/PyQt 环境不兼容且无法修改，因此界面层与推理层被重写为纯 Python 实现。
> 重写过程中**没有复制、反编译或再分发**该加密组件的任何代码。

下面列出的部分来自第三方，按各自的许可使用，特此声明。

## 1. YOLOv5（Ultralytics）

- 位置：同级目录 `PlantDocDetection/models/`、`PlantDocDetection/utils/`（`datasets.py`、`general.py`、
  `torch_utils.py` 等），通过 `YOLO_REPO` 环境变量或默认的同级路径引用。
- 许可：**GPL-3.0**（YOLOv5 v5.0 采用 GPL-3.0；后续版本改为 AGPL-3.0）。
- 使用范围：本项目只调用其 `attempt_load`、`letterbox`、`non_max_suppression`、
  `scale_coords`、`check_img_size`、`select_device` 等推理期函数，未修改其内部实现。
- 影响：**分发**本项目（含打包成 exe 发给别人）时，需要一并提供 YOLOv5 的源码和许可声明，
  并且整个作品的许可需要兼容 GPL-3.0。仅自己运行、或在论文/作业中演示不触发分发义务，
  但仍应在论文与 README 中注明来源。

## 2. PlantDoc 数据集

- 出处：Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., Batra, N.
  *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020.
- 本仓库使用的标注为 YOLO 格式导出（文件名带 `.rf.` 哈希，由 Roboflow 导出生成）。
- 许可：请以数据集官方发布页的声明为准（学术研究用途）。论文中引用上述文献即可。

## 3. 预训练权重 `weights/plant-best.pt`

- 由 PlantDoc 数据集在 YOLOv5 上微调得到。权重的训练代码属于 YOLOv5（GPL-3.0），
  因此权重的分发同样受其约束。
- 如果权重来自他人训练产物，请在 README 中如实注明"权重来源"，
  这也是答辩时通常会被追问的一点。

## 4. 未使用的部分（重要）

本项目的 `leafscan/` **没有复用**上游的界面代码。`PlantDocDetection/utils/__init__.py`
是 PyArmor 加密的商业授权组件，它提供的 `QMainWindow` 基类、`drawRectBox`、
`display_image` 等界面封装属于第三方闭源代码；`leafscan` 的界面层（`ImageView` +
`MainWindow` + `QPainter` 叠加绘制 + 自建信号槽）全部自行实现，不存在对该加密组件的
复制、反编译或再分发。

**但有一件事必须说清楚**：这个加密文件在运行期**确实会被加载**，只是不参与业务逻辑。
`import utils.general` 会先执行 `utils/__init__.py`，它再经 PyArmor 运行时
（`utils/engine/py38/...`）导入 `PlantDocDetection_UI.py` 与 `RecSystem_rc.py`。
实测一次完整推理，落在上游目录的模块共 13 个，上述三个都在其中（清单见 README §2.3）。
删掉会直接 ImportError，所以处理方式是**原样保留、不改不删**：把上游目录当作
不可修改的第三方依赖引用，性质与 `venv38/Lib/site-packages` 里的包相同。

若运行时与当前 Python 版本不匹配，请改用与上游一致的 Python 版本（见 README 第 2 节），
或把 `detector.py` 里这几处 import 换成自己复制的纯 Python 实现（YOLOv5 许可允许，
但需保留 GPL 声明）。

## 5. 整体许可状态（分发前必看）

- 本仓库根目录的 `LICENSE` 是 **GPL-3.0 全文**，仓库整体（含 `leafscan/`、`tools/` 这些作者编写的代码）
  按 GPL-3.0 授权：作者代码本身没有借用上游实现，但它在**运行时 import YOLOv5 的模块**
  （`models.experimental.attempt_load`、`utils.general.letterbox` 等），使用的是
  由 YOLOv5 训练得到、随 GPL-3.0 发布的权重。这种链接方式使整合作品在**对外分发时**
  必须整体按 GPL-3.0 提供源码，并保留上游的版权声明。
- "分发"指把打包后的 exe / 文件夹交给别人。仅本机运行、论文演示、答辩现场运行
  **不触发**分发义务；但无论是否分发，都应在文档中注明上游来源（本文件即为此用途）。
- 交付时的最小合规包：`LICENSE`（GPL-3.0 全文）+ `NOTICE.md`（本文件）+ 上游源码获取地址
  （<https://github.com/ultralytics/yolov5>，注意区分 v5.0 的 GPL-3.0 与后续版本的 AGPL-3.0）。
- 本仓库**不包含也不分发** `PlantDocDetection/utils/__init__.py` 这个加密组件。
- `weights/plant-best.pt` 的训练代码属于 YOLOv5，因此权重的再分发同样受 GPL-3.0 约束；
  如果权重来自他人，请在 README 中注明来源。

## 6. 结果可复现性

本仓库报告的所有数字都可由下列命令重新生成，且不依赖本文件以外的私有材料：

```bat
python tools\smoke_test.py     :: 端到端冒烟测试
python tools\evaluate.py       :: 全量 239 张测试图 → docs/eval_report.md（约 60 s，CPU）
```

评估口径为 VOC 单阈值（IoU=0.5，all-point 插值），不是 COCO 的 mAP@[.5:.95]，
引用时请连同口径一起说明。
