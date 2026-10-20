# -*- coding: utf-8 -*-
"""LeafScan：叶片病害检测工作台。

模块划分：
    detector  —— 纯推理层，不依赖 Qt，可被脚本/服务复用
    workers   —— 后台线程，保证界面不被推理阻塞
    workbench —— 界面层，只做展示与交互
"""

import os
import tempfile
from pathlib import Path

__version__ = "1.0.0"


def _ensure_ascii_temp() -> None:
    """把临时目录切到纯 ASCII 路径。

    起因：YOLOv5 的 utils/general.py 顶部 import scipy.signal，而 scipy 的
    messagestream 会用 tempfile 建临时文件；Windows 用户名含中文时
    （C:\\Users\\<用户名>\\AppData\\Local\\Temp\\scipy-xxxx）它在 C 层按字节打开路径，
    直接抛 OSError: Failed to open file，整个 import 链崩掉。

    这个函数在包 __init__ 里执行，早于任何子模块 import 第三方库，
    所以还来得及把 tempfile 的查找路径改掉。
    """
    try:
        current = tempfile.gettempdir()
    except Exception:
        return
    if current.isascii():
        return

    fallback = Path(os.environ.get("SystemDrive", "C:") + "\\") / "LeafScanTemp"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        probe = fallback / "probe.tmp"
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError:
        return  # 没有写权限就保持原样，让上层自己报错

    os.environ["TEMP"] = str(fallback)
    os.environ["TMP"] = str(fallback)
    tempfile.tempdir = str(fallback)


_ensure_ascii_temp()
