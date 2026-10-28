"""Editor de video estilo TouchDesigner — punto de entrada.

Uso:  python main.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _comprobar_dependencias():
    faltan = []
    for modulo, paquete in (("cv2", "opencv-python"), ("numpy", "numpy"), ("PIL", "Pillow")):
        try:
            __import__(modulo)
        except ImportError:
            faltan.append(paquete)
    if faltan:
        print("Faltan librerías obligatorias. Instálalas con:\n  pip install " + " ".join(faltan))
        sys.exit(1)
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("Falta Tkinter. En Linux: sudo apt install python3-tk   (en Windows/macOS viene con Python).")
        sys.exit(1)
    opcionales = []
    for modulo, texto in (("ultralytics", "detección de objetos (pip install ultralytics)"),
                          ("sounddevice", "sonido en la vista previa y micrófono (pip install sounddevice)"),
                          ("imageio_ffmpeg", "audio, exportación y datamosh real sin instalar FFmpeg (pip install imageio-ffmpeg)"),
                          ("screeninfo", "elegir el monitor del proyector (pip install screeninfo)")):
        try:
            __import__(modulo)
        except Exception:
            opcionales.append(texto)
    if opcionales:
        print("Opcionales no instaladas (el resto funciona igual):")
        for o in opcionales:
            print("  -", o)


if __name__ == "__main__":
    _comprobar_dependencias()
    from ui.app import App
    App().mainloop()
