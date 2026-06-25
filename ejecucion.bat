@echo off
setlocal

REM ================================================================
REM Dashboard Intrak - Ejecucion automatica (Backend + Frontend)
REM ================================================================

set "ROOT=%~dp0"
set "PYTHON_EXE=C:\Users\kcifu\miniconda3\envs\dashboard\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] No se encontro Python del entorno dashboard:
    echo         %PYTHON_EXE%
    echo.
    echo Ajusta la ruta en ejecucion.bat y vuelve a intentar.
    pause
    exit /b 1
)

echo Iniciando servicios del dashboard...
echo.

REM Backend FastAPI (puerto 8000)
start "Intrak Backend (8000)" cmd /k "cd /d "%ROOT%" && "%PYTHON_EXE%" backend\main.py"

REM Frontend local (puerto 3000)
start "Intrak Frontend (3000)" cmd /k "cd /d "%ROOT%frontend" && "%PYTHON_EXE%" server.py"

REM Esperar unos segundos y abrir dashboard
timeout /t 3 /nobreak >nul
start "" "http://localhost:3000"

echo.
echo Servicios iniciados:
echo - Backend:  http://localhost:8000
echo - Frontend: http://localhost:3000
echo.
echo Puedes cerrar esta ventana. Las otras dos quedan ejecutando los servidores.

exit /b 0
