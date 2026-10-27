@echo off
setlocal
cd /d "%~dp0"
set PYTHONHOME=
set _ROOT=%~dp0
for %%i in ("%_ROOT%..") do set _PROJ=%%~fi
set PATH=%_PROJ%\python;%_PROJ%\python\Scripts;%PATH%
echo ========================================================
echo  HAR Deep Vision — Training on UCF101 Data
echo  Note: On CPU (no CUDA GPU), expect ~1-3 min per epoch
echo  with the small 6-class subset.
echo ========================================================
python scripts/train.py --config backend/config.yaml --epochs 30 --batch-size 4
pause
