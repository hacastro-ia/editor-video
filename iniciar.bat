@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Creando el entorno virtual...
    python -m venv .venv
    if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
if not exist ".venv\instalado.txt" (
    echo Instalando librerias. La primera vez puede tardar varios minutos...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 goto error
    echo ok> ".venv\instalado.txt"
)
python main.py
if errorlevel 1 goto error
goto fin
:error
echo.
echo Ocurrio un error. Revisa la GUIA_INSTALACION.md (seccion de problemas frecuentes).
pause
:fin
