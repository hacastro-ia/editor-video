#!/usr/bin/env bash
# Crea el entorno virtual e instala las librerías la primera vez; después solo abre el programa.
cd "$(dirname "$0")" || exit 1
if [ ! -d .venv ]; then
  echo "Creando el entorno virtual..."
  python3 -m venv .venv || { echo "No se pudo crear el entorno (¿falta python3-venv?)"; exit 1; }
fi
source .venv/bin/activate
if [ ! -f .venv/instalado.txt ]; then
  echo "Instalando librerías. La primera vez puede tardar varios minutos..."
  python -m pip install --upgrade pip && pip install -r requirements.txt && touch .venv/instalado.txt \
    || { echo "Error instalando las librerías. Revisa GUIA_INSTALACION.md"; exit 1; }
fi
python main.py
