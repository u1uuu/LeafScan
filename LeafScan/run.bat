@echo off
rem 固定用仓库自带 venv38 里的解释器启动，避免落到系统 Python 3.11（没装 torch/numpy）
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\venv38\Scripts\python.exe"
if not exist "%PY%" (
    echo [LeafScan] 找不到 venv38 解释器：%PY%
    echo [LeafScan] 请按 README 第 2 节创建虚拟环境，或把 run.bat 里的 PY 改成你的路径。
    echo [LeafScan] 先用哪个 python 自检：python tools\doctor.py
    pause
    exit /b 2
)

echo [LeafScan] interpreter: %PY%
"%PY%" run_leafscan.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [LeafScan] 运行失败，退出码 %RC%，请检查上面的报错信息。
    pause
)
endlocal & exit /b %RC%
