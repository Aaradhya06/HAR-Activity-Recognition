@echo off
setlocal
cd /d "%~dp0"
set PYTHONHOME=
set _ROOT=%~dp0
for %%i in ("%_ROOT%..") do set _PROJ=%%~fi
set PATH=%_PROJ%\python;%_PROJ%\python\Scripts;%PATH%
echo ========================================================
echo  HAR Deep Vision — Local Fine-Tuning (CPU)
echo  Trains on all 24 videos with frozen backbone
echo  Expect ~15-20 minutes. Check logs for accuracy.
echo ========================================================
python scripts/finetune_alldata.py --epochs 80 --lr 0.001 --clips-per-video 15
pause
