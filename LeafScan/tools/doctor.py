# -*- coding: utf-8 -*-
"""环境自检：把"点不开"拆成缺包 / 缺权重 / 解释器不对三种情况分别报出来。

用法（在 LeafScan/ 目录下）：
    ..\\venv38\\Scripts\\python.exe tools\\doctor.py
    python tools\\doctor.py          # 若当前 python 已激活 venv38
"""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))

import leafscan  # noqa: E402,F401  （导入即触发临时目录修正，先于任何第三方库）

OK = "[ OK ]"
NG = "[FAIL]"


def check_module(name: str):
    """返回 (标记, 描述)。"""
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:  # 半损坏的安装
        return NG, f"{name}: 查找失败 -> {type(exc).__name__}: {exc}"
    if spec is None:
        return NG, f"{name}: 未安装"
    try:
        module = __import__(name)
    except Exception as exc:
        return NG, f"{name}: 已安装但导入失败 -> {type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "")
    if name == "PyQt5":  # PyQt5 没有 __version__，版本在 QtCore 里
        try:
            from PyQt5.QtCore import PYQT_VERSION_STR

            version = PYQT_VERSION_STR
        except Exception:
            version = "?"
    return OK, f"{name}: {version}".rstrip()


def resolve_upstream():
    """按 detector.py 的规则推导上游目录，但**不 import detector**。

    detector 在模块顶层就执行 `from models.experimental import attempt_load`，
    上游缺失时那行直接抛 ModuleNotFoundError；而自检要做的恰恰是把"缺文件"
    当成一条可读的检查项，所以这里复现一遍同样的路径规则（环境变量 YOLO_REPO
    优先，否则取 <仓库根>/PlantDocDetection）。
    """
    raw = os.environ.get("YOLO_REPO") or str(ROOT.parent / "PlantDocDetection")
    upstream = Path(raw).resolve()
    return upstream, upstream / "weights" / "plant-best.pt"


def main() -> int:
    print("=== LeafScan 环境自检 ===")
    print(f"解释器   ：{sys.executable}")
    print(f"版本     ：{sys.version.split()[0]}")
    print(f"平台     ：{platform.platform()}")
    print(f"工作目录 ：{os.getcwd()}")
    print(f"临时目录 ：{tempfile.gettempdir()}")
    if not str(Path.home()).isascii():
        print("           （家目录含非 ASCII 字符；leafscan/__init__.py 已把临时目录切到 C:\\LeafScanTemp，上面是改后的值）")
    print()

    dep_failures = 0

    print("--- 依赖 ---")
    for mod in ("numpy", "cv2", "torch", "PyQt5", "yaml"):
        tag, msg = check_module(mod)
        print(f"{tag} {msg}")
        if tag == NG:
            dep_failures += 1
    print()

    upstream, weights = resolve_upstream()
    upstream_failures = 0

    print("--- 上游 YOLOv5 与权重 ---")
    if dep_failures:
        print("[跳过] 依赖不全，先装依赖再看这一节")
    else:
        items = [
            ("YOLOv5 源码根目录", upstream),
            ("models/experimental.py", upstream / "models" / "experimental.py"),
            ("utils/general.py", upstream / "utils" / "general.py"),
            ("权重 plant-best.pt", weights),
        ]
        for label, path in items:
            exists = Path(path).exists()
            print(f"{OK if exists else NG} {label}: {path}")
            if not exists:
                upstream_failures += 1
        print()
        print("提示：上游目录默认取 <仓库根>/PlantDocDetection，可用环境变量 YOLO_REPO 覆盖。")

    print()
    failures = dep_failures + upstream_failures
    if failures:
        print(f"自检结果：{failures} 项失败。")
        if dep_failures:
            print("缺 numpy / torch / cv2 时，几乎一定是解释器用错了——")
            print("本仓库的依赖装在 venv38 里，请用 run.bat 启动，或指定：")
            print(f'  "{ROOT.parent / "venv38" / "Scripts" / "python.exe"}" run_leafscan.py')
        if upstream_failures:
            print("缺上游文件：本仓库只含应用层，PlantDocDetection/（YOLOv5 源码 + 权重）需要自备——")
            print(f"  1) 把它放到 {upstream}（与 LeafScan/ 同级）")
            print("  2) 或设置环境变量指向已有位置：set YOLO_REPO=D:\\path\\to\\PlantDocDetection")
            print("  详见 README §2.0。")
        return 1

    print("自检结果：全部通过，可以启动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
