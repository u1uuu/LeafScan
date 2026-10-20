# LeafScan 叶片病害检测工作台

基于 YOLOv5 + PlantDoc 数据集的桌面端植物叶片病害检测应用。界面用 PyQt5，
推理走后台线程，支持图片、文件夹批量、视频与摄像头四种输入。

## 1. 功能

- 单张图片检测
- 整个文件夹批量检测（递归子目录，状态栏显示进度）
- 视频文件检测（逐帧取帧 → 提交 → 异步回显）
- 摄像头实时检测
- 类别筛选：只显示某一类，**不重新推理**，仅过滤已有结果
- 结果表格与图上框联动：点表格某行，对应框高亮
- 结果导出（把框烧进图片）
- 换模型（重新加载权重时界面不卡）
- 命令行批量模式，可在没有显示器的环境跑

## 2. 环境与运行

> **先读这一段：本仓库不含上游，clone 下来不能直接跑。**
> `PlantDocDetection/`（YOLOv5 v5.0 推理源码 + 一个加密的界面组件）、权重 `plant-best.pt`
> 和 PlantDoc 数据集**都不在本仓库**：前者是 GPL-3.0 的上游代码，权重是第三方训练产物，
> 数据集另有学术用途许可（三者关系见 `NOTICE.md`）。本仓库只包含 `LeafScan/` 这一层应用代码。
> 缺上游时 `tools/doctor.py` 会把它当普通缺文件报出来，不会抛一堆 traceback。

### 2.0 前置条件：把上游放回去

最省事的做法是把**整个 `PlantDocDetection/` 原样放回同级位置**（约 105 MB / 569 个文件）：

```
<仓库根>/
├─ LeafScan/                  ← 本仓库，clone 下来的就是这一层
└─ PlantDocDetection/         ← 需要自备
   ├─ models/                   YOLOv5 v5.0：common.py / experimental.py / yolo.py
   ├─ utils/                    letterbox / NMS / scale_coords + PyArmor 运行时 utils/engine/py38/
   ├─ weights/plant-best.pt     PlantDoc 微调权重
   ├─ PlantDocDetection_UI.py   运行期会被 import（加密组件，见 NOTICE.md §4）
   ├─ RecSystem_rc.py           同上
   └─ PlantDoc/                 images/test + labels/test，只有 tools/evaluate.py 需要
```

- 上游也可以放在任意位置，用环境变量指过去，不必和 `LeafScan/` 同级：
  `set YOLO_REPO=D:\third_party\PlantDocDetection`（`detector.py` 第 27 行读取）。
- 界面与 CLI 只需要 `models/`、`utils/`、`weights/`、两个顶层 `.py`；`PlantDoc/` 缺失只影响评估脚本。
- §2.3 记录了哪些子目录可以裁、裁之前怎么验证指标没变（`PlantDocDetection/` 由 576 MB 裁到 105 MB）。
- 权重来自 PlantDoc 数据集在 YOLOv5 上的微调产物，**不是本仓库训练的**；引用时请连同来源一起标注。

装好依赖后启动（`requirements.txt` 已包含上游 `utils/general.py` 需要的 scipy / matplotlib / Pillow / tqdm）：

```bash
pip install -r requirements.txt
python run_leafscan.py
```

权重默认取同级的 `PlantDocDetection/weights/plant-best.pt`，也可以指定：

```bash
python run_leafscan.py --weights D:\models\plant-best.pt --img-size 640
python run_leafscan.py --cli D:\imgs --save-dir D:\out   # 无界面批量
```

**Python 版本**：上游 `PlantDocDetection/utils/__init__.py` 是 PyArmor 加密文件，
它的运行时按解释器版本和平台打包在 `utils/engine/` 下。本机该目录里有
`py37 / py38 / py39 / py310 / py311` 五套，所以 3.7–3.11 都能跑；
已在 `Python 3.8 + torch 1.8.0+cpu` 的虚拟环境上实测通过（虚拟环境不入库，自己建一个并 `pip install -r requirements.txt` 即可）。

> **这台机器上其实有三个 Python**（`py -0p` 可以列出来）：
>
> | 解释器 | 依赖 | 什么时候会被选中 |
> | --- | --- | --- |
> | `venv38\Scripts\python.exe`（3.8.10） | ✅ numpy 1.22 / opencv 4.5 / torch 1.8.0+cpu / PyQt5 | 只在**已经激活** venv 的终端里 |
> | 系统 `Python 3.11`（`python` 默认指向它） | ❌ 连 numpy 都没有 | 新开的终端、VS Code 默认解释器 |
> | 系统 `Python 3.8` | ❌ 同上 | `py -3.8` |
>
> 症状：明明刚才还能跑，重开一个终端就变成
> `ModuleNotFoundError: No module named 'numpy'`（或 `'torch'`）。
> **不是代码坏了，是 `python` 落到了没装依赖的那个解释器上。**
>
> 三个解法，任选其一：
> 1. 双击 `run.bat`——内部固定 `..\venv38\Scripts\python.exe`，找不到会直接报错，不会静默降级到系统 Python
> 2. VS Code 里按 **F5** 选 **"LeafScan (GUI)"**——`.vscode/launch.json` 已指定解释器
> 3. 手动写全路径：`..\venv38\Scripts\python.exe run_leafscan.py`
>
> `run_leafscan.py` 启动前会做一次依赖预检：缺 numpy/cv2/torch 时不抛原始 traceback，
> 而是打印当前解释器和应该用的解释器，退出码 2。

### 2.1 环境自检

```bat
..\venv38\Scripts\python.exe tools\doctor.py          :: 解释器 / 依赖 / 上游路径 / 权重，逐项检查
..\venv38\Scripts\python.exe tools\smoke_test.py      :: 模型加载 + 推理 + 坐标校验 + 导出绘制
..\venv38\Scripts\python.exe tools\smoke_test.py --gui  :: 额外用离屏模式构建主窗口
```

`doctor.py` 的本机实测输出（全部 `[ OK ]`）：解释器 `venv38\Scripts\python.exe`、Python 3.8.10、
numpy 1.22.3 / cv2 4.5.5 / torch 1.8.0+cpu / PyQt5 / yaml 6.0，
上游根目录与 `weights\plant-best.pt` 均存在。任何一项 `[FAIL]` 都会给下一步的修复指令。

### 2.2 编辑器里的黄色波浪线（Pylance）不影响运行

在 `leafscan\detector.py` 里会看到：

```
无法解析导入 "models.experimental"  Pylance(reportMissingImports)
无法解析导入 "utils.datasets"       Pylance(reportMissingImports)
```

这是**静态分析**看不见运行时行为导致的：第 34–35 行在 import 之前把
`..\PlantDocDetection` 插进了 `sys.path`，Pylance 不会执行这段代码，所以不知道
`models` / `utils` 在哪。**程序照常运行**（`smoke_test.py` 能证明）。

要让黄线消失，本仓库已经做了两件事：

- 根目录 `pyrightconfig.json`：`extraPaths` 指向 `PlantDocDetection`，`venv` 指向 `venv38`
- 根目录 `.vscode/settings.json`：`python.defaultInterpreterPath` 指向
  `venv38\Scripts\python.exe`，并按同一份 `extraPaths` 配了 `python.analysis.extraPaths`

改完需要重载窗口（`Ctrl+Shift+P` → `Developer: Reload Window`）才会生效。

本机实测输出（CPU，img_size=480）：

```
权重  ：...\PlantDocDetection\weights\plant-best.pt
设备  ：cpu  half=False  img_size=480
类别数：30
推理  ：248.3 ms，目标 2 个
  番茄叶斑病        0.856  (63, 204, 392, 396)
  番茄叶斑病        0.735  (54, 460, 234, 623)
坐标范围检查：通过
导出绘制检查：通过
MainWindow 构建：通过，表格 2 行
ALL OK
```

CLI 模式实测（3 张 PlantDoc 测试图）：

```
%2320+...Speck_jpg.rf.529f...jpg: 2 个目标 / 326 ms / 番茄叶斑病×2
0000_jpg.rf.91a7...jpg: 1 个目标 / 307 ms / 葡萄叶黑腐病×1
000_jpg.rf.db0f...jpg: 2 个目标 / 196 ms / 甜椒叶×1, 甜椒叶斑病×1
共 3 张，检出 5 个目标
```

### 2.3 `PlantDocDetection/` 的依赖边界：哪些不能删，哪些能删

**不能整个删。** 它是运行期硬依赖——`leafscan/detector.py` 只负责「读图 → letterbox → 前向 →
NMS → 坐标逆映射 → 画框」，真正的前处理/后处理/网络结构都在上游 `models/`、`utils/` 里。
删掉后第一行就变成 `ModuleNotFoundError: No module named 'models'`，整个界面起不来；
权重也放在 `PlantDocDetection/weights/plant-best.pt`。

用一次真实推理（加载模型 + `infer()`）把 `sys.modules` 里落在上游目录的模块全列出来，
实际只加载这些文件：

| 文件 | 作用 |
| --- | --- |
| `models/{__init__,common,yolo,experimental}.py` | 网络结构 + `attempt_load` |
| `utils/__init__.py` | PyArmor 加密的界面封装，`import utils.*` 会先执行它 |
| `utils/engine/__init__.py` + `engine/py38/windows_x86_64/pyarmor_runtime.pyd` | 上面那个加密件的运行时（py37–py311 各一套，按解释器版本取） |
| `utils/{datasets,general,google_utils,torch_utils}.py` | `letterbox` / `non_max_suppression` / `scale_coords` / 设备选择 |
| `PlantDocDetection_UI.py`、`RecSystem_rc.py`（14.1 MB） | 被加密的 `utils/__init__.py` 反向导入。**看着像老界面，其实在加载链上，删了就崩** |
| `weights/plant-best.pt`（14.2 MB） | 权重 |

**实测验证**：把上面几项（`models/`、`utils/`、`data/`、`weights/`、两个 UI 的 `.py`）单独复制成
**61.8 MB** 的目录，用 `YOLO_REPO` 指过去，`run_leafscan.py --cli` 和 `smoke_test.py --gui`
都返回 0，检测框与用原目录时逐位一致 → 其余部分 LeafScan **确实用不到**。

| 已于 2026-10-20 删除（LeafScan 不读） | 体积 |
| --- | --- |
| `PlantDoc/images/train` + `labels/train`（2330 图 / 2330 标注） | 400.1 MB |
| `PlantDoc/logs` | 29.9 MB |
| `icons/` | 10.3 MB |
| `test_/` | 12.4 MB |
| 上游 `tools/`（2 个脚本） | 3.9 MB |
| `runs/`、`.idea/`、`__pycache__/` | 3.6 MB |
| 根目录重复的 `yolov5s.pt`（`weights/yolov5s.pt` 保留） | 14.5 MB |
| **合计** | **≈474 MB** |

删除流程（想复现就用这个顺序，别直接 `Remove-Item`）：

1. 先把目标目录 `Move-Item` 到临时隔离目录（同盘是重命名，瞬间完成，随时能移回）；
2. 跑 `smoke_test.py --gui` 和 `evaluate.py --limit 20`；
3. 指标与删除前**逐位一致**，才真正删除隔离区。

> 坑：`images/train` 里的文件名由 Roboflow 按图片标题拼出来，最长的一批**超过 260 字符**。
> PowerShell 的 `Remove-Item -Recurse` 会静默跳过这 14 个（残留 4.8 MB），
> 要用 `cmd /c rd /s /q \\?\<绝对路径>` 才能删干净。`Move-Item` 移走整个目录不受影响。

本次实测：`PlantDocDetection/` 从 **576.3 MB / 5298 文件** 降到 **104.8 MB / 569 文件**，
删除前后 mAP@0.5 都是 **0.7822**（20 张抽样），P/R/F1、TP/FP/FN、逐类 AP 全部相同，`ALL OK`。
想重新具备训练能力时，从 PlantDoc 官方发布重新下载 train split 即可——`plantdoc.yaml`、`train.py`
和 `train.txt` 都还在，路径不变。

**两件不能顺手删的事**：

1. `PlantDoc/images/test`（38.5 MB）+ `labels/test` **要留着**：`tools/evaluate.py` 默认从这里读
   239 张图和标注，`docs/eval_report.md` 里 mAP@0.5 = 0.673 的可复现性靠它；
   `smoke_test.py` 也是从上游目录 glob 一张图来跑冒烟测试。
2. 发 GitHub 时 `PlantDocDetection/` **不要进仓库**：它是 GPL-3.0 的上游代码 + 第三方权重，
   混进去会让整个仓库的许可变复杂（见 `NOTICE.md` §5）。仓库根的 `.gitignore` 已经把它、
   `venv38/`、`离线依赖包/` 排除掉了。


## 3. 目录结构

```
LeafScan/
├─ run.bat                   双击启动（固定用 venv38 的解释器）
├─ run_leafscan.py          入口：解析参数，选 GUI 或 CLI
├─ leafscan/
│  ├─ __init__.py            版本号 + 中文临时目录修正
│  ├─ detector.py           纯推理层：无 Qt 依赖，可单测、可服务化
│  ├─ workers.py            后台线程：推理 / 模型加载 / 批量
│  ├─ workbench.py          界面层：QPainter 叠加绘制 + 信号槽
│  └─ labels_zh.yaml        类别英文名 -> 中文名（不在代码里硬编码）
├─ tools/
│  ├─ smoke_test.py          冒烟测试：模型 -> 推理 -> 坐标 -> 界面
│  ├─ doctor.py              环境自检：解释器 / 依赖 / 上游路径 / 权重
│  └─ evaluate.py            mAP / 逐类 AP / 混淆 / 阈值扫描
├─ docs/
│  └─ eval_report.md         实跑生成的评估报告（mAP@0.5 = 0.673）
├─ requirements.txt
└─ NOTICE.md                代码来源与许可（重写部分 vs 上游 YOLOv5/PlantDoc）
```

三层职责单向依赖：`workbench → workers → detector`，`detector` 不反向依赖任何界面代码。
好处是换界面（PySide6 / Web）时只需要重写 `workbench.py`，推理与线程模型原样复用。

## 4. 关键实现说明

### 4.1 letterbox 与 scale_coords 必须成对出现

YOLOv5 推理前用 `letterbox` 把原图等比缩放到 `img_size`，短边不足的地方补灰边
（pad）。所以网络输出框的坐标系是"带灰边的方形图"，不是原图。

`scale_coords(input_shape, coords, original_shape)` 干的事就是把这个坐标系反推回原图：
先减掉 pad，再除以缩放比。三处必须一致，否则框会整体偏移：

1. 传进去的 `input_shape` 必须是**真正喂给网络**的张量后两维；
2. 必须是同一帧的尺寸，不能拿上一帧的；
3. 只在 `letterbox` 之后、`NMS` 之后各做一次，不要重复调用。

本项目的写法是：

```python
letterboxed, _, _ = letterbox(frame_bgr, new_shape=self.img_size)
tensor = torch.from_numpy(...).float() / 255.0
input_shape = tensor.shape[2:]                      # 这一行就是"喂进网络"的尺寸
...
det[:, :4] = scale_coords(input_shape, det[:, :4], frame_bgr.shape).round()
```

### 4.2 两个阈值分别在管什么

- `conf_thres`（默认 0.25）：在 NMS **之前**过滤候选框，只留模型自己觉得有把握的；
- `iou_thres`（默认 0.45）：在 NMS **之内**做抑制，两个框 IoU 超过它就认为是同一个目标，
  只保留分数高的。

调参直觉：`conf` 调高 → 漏检多、误检少；`conf` 调低 → 全检率高但误检多。
`iou` 调低 → 相邻的两个目标容易被合成一个（密集叶片场景常见）；`iou` 调高 → 同一目标出现多个框。

### 4.3 线程模型：丢旧帧，而不是排队

`InferenceWorker` 用一个 `Condition` 保护"待处理帧"槽位，`submit()` 直接覆盖旧帧：

```python
def submit(self, frame, frame_id):
    with self._cond:
        self._frame = frame           # 覆盖，不是 append
        self._frame_id = frame_id
        self._cond.notify()
```

这样当推理耗时（比如 80 ms）大于取帧间隔（30 ms）时：

- 排队模型：延迟越积越长，画面越来越滞后，最后内存被帧对象撑爆；
- 丢帧模型：始终只处理最新一帧，延迟稳定在"一次推理耗时"。

界面显示的是 `detected` 信号里**回传的那一帧**，而不是主线程当前持有的帧。
因为回传的帧和框是同一份数据，永远不会出现"图是第 100 帧、框是第 99 帧"的错位。

### 4.4 信号连接：用 blockSignals，不用反复 disconnect

重建下拉框时如果用 `disconnect()` + `connect()`，一旦某次 `disconnect` 抛异常
（没连上时 Qt 会警告）或者某条路径漏了配对，就会出现一个信号触发多次槽函数。
正确做法是临时屏蔽：

```python
blocked = self.combo_filter.blockSignals(True)
...  # clear() + addItem()
self.combo_filter.blockSignals(blocked)
```

表格重建同理，否则 `setRowCount` + 逐行 `setItem` 会触发 N 次 `itemSelectionChanged`。

### 4.5 中文的两个坑

- `cv2.putText` 只支持 ASCII，中文会变成 `???`。所以界面上用 `QPainter.drawText` 画标签，
  导出文件时用 `cv2.putText` 写英文类名——保证导出的图片不乱码。
- `cv2.imread` / `cv2.imwrite` 在 Windows 上不支持中文路径，
  统一走 `np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `tofile`。

### 4.6 两个只在"Windows 中文用户名"下出现的坑（本机实测踩到）

1. **scipy 在 import 阶段就崩，整个程序起不来**。
   `utils/general.py` 顶部有 `from scipy.signal import butter, filtfilt`，
   scipy 的 `messagestream` 会用 `tempfile` 建临时文件，而默认临时目录是
   `C:\Users\<用户名>\AppData\Local\Temp`，它在 C 层按字节打开这个路径：

   ```
   OSError: Failed to open file b'C:\\Users\\<用户名>\\AppData\\Local\\Temp\\scipy-az7r52up'
   ```

   这个异常和 YOLOv5、和本项目代码都无关，纯粹是路径编码问题，但表现是"程序一点就闪退"。
   解决：`leafscan/__init__.py` 里的 `_ensure_ascii_temp()` 在导入任何第三方库之前，
   把 `TEMP` / `TMP` 和 `tempfile.tempdir` 切到 `C:\LeafScanTemp`（仅当原路径含非 ASCII 字符时）。
   这也是个很好的答辩素材：**"在我机器上能跑"不等于"能交付"**。

2. **`QFontDatabase: Cannot find font directory ...` 刷屏**。
   PyQt5 的 wheel 不自带 `Qt5/lib/fonts` 目录，Qt 每次查字体都打一行警告。
   建个空目录就安静了（不影响功能）：

   ```bash
   mkdir venv38\lib\site-packages\PyQt5\Qt5\lib\fonts
   ```

### 4.7 实测指标（本机实跑，可复现）

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| **mAP@0.5** | **0.673** | VOC 单阈值、all-point 插值、27 个类别、454 个 GT 框 |
| P / R / F1 | 0.582 / 0.659 / 0.618 | 工作点 `conf >= 0.25`（TP 299 / FP 215 / FN 155） |
| 单帧耗时 | ≈170–210 ms | CPU、`img_size=480`、含预处理与 NMS（三次实测 172 / 183 / 206 ms，随机器负载浮动） |

复现命令（在 `LeafScan/` 下执行，约 60 秒）：

```bat
python tools\evaluate.py              :: 全量 239 张 → docs/eval_report.md
python tools\evaluate.py --limit 20   :: 小样本先验证流程
python tools\smoke_test.py            :: 端到端冒烟测试（模型加载→推理→界面构建）
```

逐类 AP、类别混淆表、置信度阈值扫描见 **[`docs/eval_report.md`](docs/eval_report.md)**。
注意这是 VOC 口径的 mAP@0.5，**不是** COCO 的 mAP@[.5:.95]，数值不可直接比较。

## 5. 已知限制

1. **推理后端只支持 PyTorch**。同一份 `detector.py` 换成 ONNX Runtime 或 TensorRT
   可以再快 2–4 倍，接口不用改，只替换 `infer()` 内部实现。
2. **实时流与批量不并发**。两者共享同一个 `Detector`（省一次显存），
   而 PyTorch 模块在多个线程同时前向时并不安全，所以批量期间界面会禁用其它入口。
   要真并发就得上多进程或每进程一份模型。
3. **没有目标跟踪**。摄像头模式下每帧独立检测，同一片叶子在两帧里是两个独立框。
   加 IoU 匹配或 ByteTrack 就能补上 id 保持。
4. **评估口径是 VOC 单阈值**。`tools/evaluate.py` 已实现并跑完全部 239 张测试图
   （mAP@0.5 = 0.673，见 `docs/eval_report.md`），但没有 COCO 的 mAP@[.5:.95]；
   需要那套口径得接 pycocotools，并先把 YOLO 标注转成 COCO json。
5. **界面语言固定中文**，没有做 i18n。
6. 摄像头固定用设备 0，没有设备选择。

## 6. 设计问答（答辩常问）

**关于预处理与后处理**

1. `img_size` 为什么必须是 32 的倍数？
   → YOLOv5 的 `stride` 最大是 32（P3/P4/P5 三个输出层分别下采样 8/16/32 倍），
   输入尺寸不能被 stride 整除时特征图对不齐。`check_img_size` 就是向上取整到 stride 的倍数。

2. 为什么要 `letterbox` 而不是直接 `resize`？
   → 直接 resize 会改变长宽比，目标形状被拉伸，和训练时的数据分布不一致，掉点。
   `letterbox` 保持比例、补灰边，代价是有一部分算力花在灰边上。

3. 一张 480×480 的图，网络输出是什么形状？
   → 三个尺度：(60×60 + 30×30 + 15×15) × 3 个 anchor = 14175 个预测，
   每个预测是 `5 + num_classes` 维（中心点 xy、宽高、置信度、类别分数）。

4. `conf_thres` 和 `iou_thres` 分别作用在哪个阶段？
   → 见 4.2。一个是"分数够不够"，一个是"两个框重不重叠"。两者都过低会把背景检成目标。

**关于工程实现**

5. 为什么推理要放 `QThread`，不能直接用 `QTimer` + `processEvents()`？
   → `processEvents()` 会把用户输入、重绘事件重入到正在推理的调用栈里，
   可能触发两次"打开视频"从而导致两个定时器同时喂帧；而且主线程不返回，
   窗口会显示"未响应"。正确做法是工作线程 + 信号槽回主线程。

6. 待处理帧为什么只留一帧？
   → 见 4.3。排队会让延迟单调增长；丢旧帧让延迟恒定。

7. `QImage` 为什么要 `.copy()`？
   → `QImage` 直接包 numpy 内存时并不持有它，`frame` 被 GC 或复用后画面会花屏/崩溃。

8. 为什么批量模式要把 `item_done` 里的 frame 一起发出来？
   → 保证"显示的行"和"显示这行的图"是同一份数据；重新读文件既慢又可能读到被改动的文件。

**关于指标与改进**

9. mAP@0.5 和 mAP@0.5:0.95 的区别？
   → 前者只在 IoU=0.5 判定下算 AP（"有没有框住"），后者在 0.5~0.95 每 0.05 取一次平均
   （"框得准不准"）。定位精度差时后者掉得很快。

10. PlantDoc 这种数据集有什么固有难点？
    → 类别不均衡（番茄类多、覆盆子/蓝莓少）；同一张图多片叶子 → 密集小目标；
    病害区域是叶面局部，框往往很大而病斑很小，模型容易只学到"这是番茄叶"而不是"这是病"。

11. 想提升小目标召回，成本最低的改法？
    → 提高 `img_size`（480→640）；或在 `data/*.yaml` 里保留 P3 层特征；
    或对训练集做 mosaic 增强（上游已开）。

## 7. 后续可以做的事

- [ ] `export.py` 导出 ONNX，`detector.py` 增加 `backend="onnx"` 分支
- [x] `evaluate.py`：遍历验证集，输出混淆矩阵与每类 AP（→ `docs/eval_report.md`）
- [ ] 把阈值扫描结果接成界面上的滑条（`conf` 实时可调，滑块旁显示 F1 曲线）
- [ ] 视频结果导出为 mp4（`cv2.VideoWriter`）
- [ ] 多摄像头设备选择
- [ ] 单元测试：`letterbox` + `scale_coords` 的往返一致性（给定 box，映射后再逆映射应回到原坐标）
