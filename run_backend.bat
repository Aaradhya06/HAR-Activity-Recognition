@echo off
setlocal
cd /d "%~dp0"
set PYTHONHOME=
:: Resolve the Python and Scripts directories relative to the project's parent
set _ROOT=%~dp0
for %%i in ("%_ROOT%..") do set _PROJ=%%~fi
set PATH=%_PROJ%\python;%_PROJ%\python\Scripts;%PATH%
echo ========================================================
echo  HAR Deep Vision — Backend Server
echo  URL: http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo ========================================================
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
