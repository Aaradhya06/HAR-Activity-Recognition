@echo off
setlocal
cd /d "%~dp0"
set PYTHONHOME=
set _ROOT=%~dp0
for %%i in ("%_ROOT%..") do set _PROJ=%%~fi
set PATH=%_PROJ%\python;%_PROJ%\python\Scripts;%PATH%
echo ========================================================
echo  HAR Deep Vision — Running Pytest Test Suite
echo ========================================================
python -m pytest tests/ -v --tb=short
pause
