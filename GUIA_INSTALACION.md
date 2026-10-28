# Guía de instalación — Editor de video estilo TouchDesigner

Esta guía asume que **no tienes experiencia con Python**. Funciona en Windows, macOS y Linux.
Tiempo estimado: 15–30 minutos (la mayor parte es esperar descargas).

## Resumen en 5 pasos

1. Instala **Python 3.11 o 3.12**.
2. Descomprime el proyecto en una carpeta **sin tildes, ñ ni espacios** (ej. `C:\editor_video`).
3. Haz doble clic en **`iniciar.bat`** (Windows) o ejecuta **`./iniciar.sh`** (macOS/Linux).
   La primera vez crea un entorno e instala todo solo; después solo abre el programa.
4. Abre un video, pulsa **Reproducir** y prueba un **Preset**.
5. Si algo falla, ve a [Problemas frecuentes](#problemas-frecuentes).

Si prefieres entender y hacer cada paso a mano, sigue leyendo.

---

## 1. Qué necesitas

| | Mínimo | Recomendado |
|---|---|---|
| Python | 3.10 | **3.11 o 3.12** (las versiones muy nuevas a veces aún no tienen todas las librerías) |
| Memoria RAM | 8 GB | 16 GB |
| Disco libre | 3 GB | 6 GB (PyTorch y los modelos pesan) |
| Tarjeta gráfica | no hace falta | NVIDIA (hace fluidos YOLO y MiDaS) |
| Internet | solo para instalar y descargar modelos la primera vez | |

Funciona sin GPU, pero la detección de objetos y las personas serán más lentas (baja la resolución en la barra superior).

## 2. Instalar Python

### Windows
1. Entra a <https://www.python.org/downloads/> y descarga **Python 3.12** (instalador de Windows de 64 bits).
2. Abre el instalador y **marca la casilla «Add python.exe to PATH»** (abajo). Luego «Install Now».
3. Comprueba: abre el menú Inicio, escribe `cmd`, abre «Símbolo del sistema» y escribe:
   ```
   python --version
   ```
   Debe mostrar `Python 3.12.x`. Si dice que no se reconoce, repite la instalación marcando la casilla (o prueba `py --version`).

### macOS
1. Descarga el instalador de <https://www.python.org/downloads/> (incluye Tkinter) y ábrelo.
   (Con Homebrew: `brew install python@3.12 python-tk@3.12`.)
2. Abre la app **Terminal** y comprueba: `python3 --version`.

### Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk libportaudio2
python3 --version
```

## 3. Descargar y descomprimir el proyecto

1. Descarga `editor_video.zip`.
2. Descomprímelo en una ruta **corta y sin tildes, ñ ni espacios**:
   - Windows: `C:\editor_video`
   - macOS/Linux: `~/editor_video`
   (Las rutas con caracteres especiales dan problemas con OpenCV al abrir videos.)
3. Dentro debe verse esto: `main.py`, `requirements.txt`, `iniciar.bat`, `iniciar.sh`, carpetas `core`, `effects`, `ui`…

## 4. Abrir una terminal dentro de la carpeta

- **Windows:** abre la carpeta en el Explorador, haz clic en la barra de direcciones (arriba), escribe `cmd` y pulsa Enter.
- **macOS:** clic derecho en la carpeta → «Nueva terminal en la carpeta» (o abre Terminal y escribe `cd ~/editor_video`).
- **Linux:** clic derecho dentro de la carpeta → «Abrir en terminal».

## 5. Camino rápido (recomendado)

- **Windows:** doble clic en `iniciar.bat`.
- **macOS/Linux:** en la terminal: `chmod +x iniciar.sh` (solo una vez) y luego `./iniciar.sh`.

El script crea el entorno virtual (`.venv`), instala las librerías de `requirements.txt` y abre el programa.
**La primera vez tarda varios minutos**, sobre todo por PyTorch. Las siguientes veces abre en segundos.
Si algo sale mal, el script se detiene y muestra el error.

## 6. Camino manual (paso a paso)

Con la terminal abierta dentro de la carpeta del proyecto:

**a) Crear el entorno virtual** (una «caja» aislada para las librerías de este proyecto):
```bash
python -m venv .venv          # en macOS/Linux usa python3
```

**b) Activarlo** (cada vez que abras una terminal nueva):

| Sistema | Comando |
|---|---|
| Windows (cmd) | `.venv\Scripts\activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

Cuando funciona, la línea de la terminal empieza con `(.venv)`.
En PowerShell, si dice que la ejecución de scripts está deshabilitada, ejecuta una vez:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a intentar.

**c) Actualizar pip e instalar las librerías:**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**d) Abrir el programa:**
```bash
python main.py
```

### Qué se instala y para qué sirve cada librería

| Librería | Para qué | Peso aprox. |
|---|---|---|
| `numpy`, `opencv-python`, `Pillow` | Base: leer video, efectos, interfaz | 100 MB |
| `imageio-ffmpeg` | Audio del video y exportar MP4 (trae FFmpeg, no hay que instalarlo aparte) | 40 MB |
| `sounddevice` | Sonido en la vista previa y micrófono | pequeña |
| `ultralytics` | YOLO: detección de objetos, personas por puntos, siluetas, biblioteca de imágenes. **Instala PyTorch** | 1–3 GB |
| `screeninfo` *(recomendada)* | Lista tus monitores para elegir en cuál proyectar | pequeña |
| `pygrabber` *(opcional, Windows)* | Muestra el nombre real de cada cámara en el selector | pequeña |
| `timm` *(opcional)* | Modelo de profundidad MiDaS para la nube de puntos | pequeña |

**Instalar por etapas** (si quieres ver el programa funcionando ya y dejar YOLO para después):
```bash
pip install numpy opencv-python Pillow imageio-ffmpeg sounddevice
python main.py
# más tarde:
pip install ultralytics
```
Sin `ultralytics` funcionan todos los efectos de imagen, audio, datamosh, nube de puntos, psicodélicos, la censura de caras
y el **Traqueo de puntos (HUD)**. Solo los efectos con YOLO muestran un aviso rojo pidiendo instalarlo.

### Si tienes tarjeta NVIDIA (opcional, mucho más rápido)
Antes de instalar `ultralytics`, instala PyTorch con soporte CUDA. Entra a <https://pytorch.org/get-started/locally/>,
elige tu sistema, «Pip» y «CUDA», y copia el comando que te da la página. Luego `pip install ultralytics`.
Para comprobar: `python -c "import torch; print(torch.cuda.is_available())"` debe imprimir `True`.

## 7. Primera ejecución: lo que se descarga solo

Necesita internet **solo la primera vez que uses cada función**:

| Función | Qué descarga | Dónde queda |
|---|---|---|
| Detección de objetos, biblioteca de imágenes | `yolov8n.pt` (~6 MB) | lo gestiona Ultralytics |
| Personas: puntos y esqueleto | `yolov8n-pose.pt` (~7 MB) | idem |
| Personas: siluetas | `yolov8n-seg.pt` (~7 MB) | idem |
| Censura de caras | YuNet (<1 MB) | carpeta `models/` |
| Foto de referencia en Caras | SFace (~37 MB) | carpeta `models/` |
| Nube de puntos con MiDaS | modelo vía `torch.hub` | caché de PyTorch |

Si la descarga de **YuNet** o **SFace** falla (red de empresa, universidad…), el programa usa un detector de respaldo menos
preciso. Para ponerlos a mano, descarga los archivos y guárdalos en la carpeta `models/` **con estos nombres exactos**:

- `face_detection_yunet_2023mar.onnx` →
  <https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx>
- `face_recognition_sface_2021dec.onnx` →
  <https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx>

## 8. Comprobar que todo funciona

Con el entorno activado, estos comandos deben terminar sin errores:

```bash
python -c "import cv2, numpy, PIL; print('base OK, OpenCV', cv2.__version__)"
python -c "import tkinter; print('Tkinter OK')"
python -c "import imageio_ffmpeg; print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe())"
python -c "import sounddevice; print('Audio OK')"
python -c "import ultralytics; print('YOLO OK', ultralytics.__version__)"
```

Prueba dentro del programa (usa un video corto de 10–20 segundos con música):

1. **Abrir video** → se ve la primera imagen. Pulsa **Reproducir**: debe oírse el sonido.
2. **Presets → Mandala reactivo**: la imagen debe pulsar con la música (pestaña **Audio**: las barras se mueven).
3. **Presets → Traqueo de puntos (sin YOLO)**: puntos que siguen el movimiento.
4. **Presets → Personas: esqueleto neón** (necesita `ultralytics`): esqueletos sobre las personas.
5. **Proyectar** abre la ventana de salida (puedes arrastrarla a otro monitor y pulsar `F`).
6. **Exportar MP4** → se genera un video con efectos y audio.
7. **Presets → Cinta 8mm: pausa y rebobinado (automático)**, pulsa **Reproducir** (imagen limpia), luego **Pausar** y
   **◀◀** (rebobinar): debe aparecer el ruido de cinta. En la pestaña **Cinta** puedes insertar pausas y rebobinados en el
   video exportado.

## 8b. Cámaras y proyector (para usarlo en directo)

**Cámara.** Pulsa **Cámara** (arriba) y elige la entrada en la lista; verás una vista previa de cada una (sirve para webcams, cámaras USB y
capturadoras). Puedes fijar resolución (hasta 1080p), fps y el espejo. Si tu cámara no aparece: ciérrala en otras aplicaciones (Zoom, OBS…) y pulsa
«Buscar de nuevo». macOS: Ajustes → Privacidad y seguridad → Cámara → permite **Terminal**. Windows: Configuración → Privacidad → Cámara → permite
apps de escritorio. Con una cámara de cinta Digital8 necesitas una capturadora o conexión FireWire/USB que el sistema reconozca como cámara.

**Proyector.** Conecta el proyector (modo «extender pantalla»), pulsa **Proyectar** (o `P`), arrastra la ventana al proyector y pulsa **F** para
pantalla completa. Con `screeninfo` instalado, en la pestaña **Salida** eliges el monitor de destino y «Abrir en pantalla completa» la coloca
directamente. `B` = negro, `M` = ajuste, `Esc` = salir. Para proyectar en grande sube la *Resolución de trabajo* (1280 o más) y activa *Ahorrar CPU*.

## 9. Biblioteca de imágenes (para el efecto «Biblioteca de imágenes»)

Es la «base de datos» de la que salen las imágenes que aparecen sobre lo reconocido.

1. Pulsa el botón **Biblioteca…** (barra superior): se abre la carpeta `biblioteca/`.
2. Dentro crea una subcarpeta por cada tipo de cosa y mete tus imágenes. Nombres válidos, en inglés o español:
   `person` / `persona`, `dog` / `perro`, `car` / `auto`, `cup` / `taza`, `chair`, `laptop`, `cell phone`, `book`…
   Una carpeta `cualquiera` sirve para todo lo que no tenga carpeta propia.
3. Para llenarla automáticamente con fotos de Wikimedia Commons (necesita internet):
   ```bash
   python descargar_biblioteca.py                 # clases comunes, 12 imágenes por clase
   python descargar_biblioteca.py dog cat --n 20  # solo esas clases
   ```
   Guarda `CREDITOS.txt` con autor y licencia. Revisa las carpetas y borra lo que no sirva.
4. Si una clase no tiene imágenes, el efecto usa imágenes abstractas de demostración (no falla).

## 10. Usar el editor el resto de los días

- **Windows:** doble clic en `iniciar.bat`.
- **macOS/Linux:** `./iniciar.sh`.
- **A mano:** abre la terminal en la carpeta, activa el entorno (paso 6b) y ejecuta `python main.py`.

Para **actualizar** YOLO: con el entorno activado, `pip install -U ultralytics`.
Para **desinstalar**: borra la carpeta del proyecto (todo vive dentro de ella, incluido `.venv`).

---

## Usar Visual Studio Code

El proyecto ya incluye una carpeta `.vscode/` con la configuración lista (ejecutar con **F5**, tareas e extensión recomendada).

1. Instala **Visual Studio Code** (<https://code.visualstudio.com>) y **Python 3.11 o 3.12** (ver paso 2 de esta guía).
2. Descomprime `editor_video.zip` en una ruta sin tildes ni ñ y, en VS Code, **Archivo → Abrir carpeta…** → elige `editor_video`.
3. Acepta instalar la extensión recomendada **Python** (de Microsoft). Si no aparece el aviso: pestaña Extensiones (`Ctrl+Shift+X`) → busca «Python».
4. Crea el entorno: `Ctrl+Shift+P` → **Python: Create Environment** → **Venv** → elige tu Python → marca `requirements.txt` → **OK**.
   (Equivale a los pasos 6a-6c de esta guía; tarda unos minutos.)
5. Comprueba que abajo a la derecha dice `.venv` como intérprete. Si no: `Ctrl+Shift+P` → **Python: Select Interpreter** → el que está dentro de `.venv`.
6. **Ejecuta con `F5`** (configuración «Editor de video») o abre `main.py` y pulsa ▶ arriba a la derecha. También: Terminal → `python main.py`.

Notas: la ventana del editor aparece como una aplicación normal (no dentro de VS Code). No funciona en VS Code remoto, Codespaces ni SSH
(no tienen pantalla); en WSL solo con WSLg. Si una librería falla al importar, casi siempre es porque VS Code está usando otro Python:
repite el paso 5.

---

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `python` no se reconoce (Windows) | Reinstala Python marcando «Add python.exe to PATH», o usa `py` en lugar de `python`. |
| `No module named 'tkinter'` | Linux: `sudo apt install python3-tk`. macOS con Homebrew: `brew install python-tk@3.12`. En Windows reinstala Python con la opción «tcl/tk» marcada. |
| `No module named 'cv2'` (u otra librería) | No activaste el entorno o instalaste en otro Python. Activa `.venv` y usa `python -m pip install -r requirements.txt`. |
| `pip` falla con «No matching distribution» o errores de compilación | Tu Python es demasiado nuevo o muy viejo. Instala **3.11 o 3.12**, borra la carpeta `.venv` y repite. |
| `PortAudio library not found` (Linux) | `sudo apt install libportaudio2`. |
| No se oye el sonido en la vista previa | Revisa que `sounddevice` esté instalado, que no esté marcada «Silenciar» (pestaña Audio) y la salida de audio del sistema. |
| Aviso «No se encontró FFmpeg» | `pip install imageio-ffmpeg` (con el entorno activado). |
| El video no tiene audio utilizable | Pestaña Audio → fuente «Micrófono», o usa un video con pista de audio. |
| La webcam o el micrófono no abren | Cierra otras apps que los usen. macOS: Ajustes → Privacidad y seguridad → Cámara/Micrófono → activa **Terminal**. Windows: Configuración → Privacidad → Cámara/Micrófono → permite apps de escritorio. |
| Todo va lento | Baja **Resolución** (480). Quita efectos pesados. En los efectos YOLO sube «Detectar cada N fotogramas» y usa el modelo `n` (el más pequeño). Con GPU NVIDIA instala PyTorch CUDA (paso 6). |
| Aviso rojo «No se pudo cargar YOLO…» | `pip install ultralytics` y revisa tu conexión (la primera vez baja los pesos). |
| Aviso «YuNet no disponible» | Se usa un detector de respaldo. Descarga los modelos a mano (paso 7). |
| «Biblioteca vacía: imágenes de demostración» | No es un error: añade tus imágenes con el botón **Biblioteca…** o ejecuta `descargar_biblioteca.py`. |
| El video no abre o se ve negro | Códec raro. Conviértelo a MP4 (H.264) con HandBrake o con FFmpeg. Comprueba que la ruta no tenga tildes ni ñ. |
| `iniciar.bat` se cierra al instante | Ábrelo desde una terminal (`cmd` en la carpeta → `iniciar.bat`) para leer el mensaje. |
| Exportar falla a mitad | Revisa el espacio en disco y que FFmpeg esté disponible (`imageio-ffmpeg`). |

Si nada de esto resuelve tu problema, copia el mensaje de error completo de la terminal: casi siempre indica la causa.
