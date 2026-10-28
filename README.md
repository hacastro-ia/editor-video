# 🎬 Editor de Video estilo TouchDesigner — v1.0

**Editor de escritorio para crear visuales y editar video en tiempo real**: una cadena de **48 efectos apilables** con controles finos, modulación reactiva al audio, cámaras en vivo, ventana de proyección, datamosh (simulado y **real**), CRT/VHS, psicodélicos, detección de personas y objetos (YOLO), censura facial y exportación a MP4.

## ⬇️ Descarga

👉 **[Descargar la última versión](https://github.com/hacastro-ia/editor-video/releases/latest)**

### Windows (recomendado)
- **Instalador** (`EditorVideo-Setup.exe`) → instala en `C:\Program Files\EditorVideo`, crea acceso en Inicio/escritorio y trae desinstalador.
- **Portable** (`EditorVideo.exe`) → solo copiar y ejecutar.

> El ejecutable no está firmado digitalmente: si Windows SmartScreen avisa, pulsa *Más información → Ejecutar de todas formas*.

> Requiere **Windows 10/11 (64 bits)**. Todo el motor (Python + librerías + FFmpeg) va dentro del ejecutable: no instala nada más.

## 🚀 Primeros pasos

1. Ejecutar el programa.
2. **Primera ejecución**: descarga automática de los modelos de IA (YOLO, YuNet/SFace, MiDaS). **Requiere internet** (~50 MB). Si falla la descarga, se usa un detector de respaldo.
3. `Ctrl+O` abrir video → `Espacio` reproducir → probar un **Preset**.
4. `Ctrl+E` exportar MP4.

## ✨ Qué incluye

- **Efectos de imagen**: ajuste de color, neón, glitch RGB, feedback, bloom, CRT/VHS, datamosh, psicodélicos, caleidoscopios, plasma generativo, etc.
- **Audio**: modulación de cualquier parámetro con beat / graves · medios · agudos / volumen / micrófono.
- **YOLO**: detección de objetos, personas (puntos/esqueleto), siluetas, censura facial, biblioteca de imágenes.
- **Producción**: proyección en monitor externo, cámara en vivo (hasta 1080p), Auto-VJ, guion de cinta, datamosh real vía FFmpeg.

## 📋 Requisitos

| | Mínimo | Recomendado |
|---|---|---|
| SO | Windows 10/11 x64 | — |
| RAM | 8 GB | 16 GB |
| Disco libre | 3 GB | 6 GB |
| GPU | — | NVIDIA (fluido YOLO / MiDaS) |

## 🔧 Desde el código (desarrollo)

```bash
git clone https://github.com/hacastro-ia/editor-video.git
cd editor-video
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Compilar el ejecutable (Windows)

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "EditorVideo" --collect-all torchvision main.py
```

> ⚠️ `--collect-all torchvision` es **obligatorio**: las versiones nuevas de torchvision traen la extensión C++ como `_C_stable.pyd` y sin este flag el exe falla con `operator torchvision::nms does not exist` al usar YOLO.

Instalador: compilar `EditorVideo.iss` con Inno Setup 6 (el script está en la raíz del proyecto, junto a `dist/`).

Guía completa: [`GUIA_INSTALACION.md`](GUIA_INSTALACION.md)

## ⚠️ Notas

- Versión beta en pruebas: puede haber bugs. Si encuentras uno, abre un [Issue](https://github.com/hacastro-ia/editor-video/issues).
- Los modelos de IA y la biblioteca de imágenes se guardan en `%LOCALAPPDATA%\EditorVideo\` (`models\` y `biblioteca\`), así funciona aunque el programa esté instalado en `Program Files`.
- En la versión instalada, añade tus imágenes desde **Salida → Herramientas → «Biblioteca de imágenes…»**.

## 📄 Licencias

- Motor propio: licencia del autor.
- YOLO (Ultralytics): AGPL — usar bajo esa licencia.
- PyTorch / OpenCV / FFmpeg: licencias BSD/LGPL correspondientes.
