@echo off
setlocal
cd /d "%~dp0\frontend"
set _ROOT=%~dp0
for %%i in ("%_ROOT%..") do set _PROJ=%%~fi
set PATH=%_PROJ%\node;%PATH%
echo ========================================================
echo  HAR Deep Vision — Frontend Web App
echo  URL: http://localhost:5173
echo ========================================================
npm run dev
pause
