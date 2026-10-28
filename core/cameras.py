"""Detección de cámaras y captura en un hilo aparte (baja latencia para usar el programa en directo)."""
from __future__ import annotations

import glob
import os
import sys
import threading
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

RESOLUCIONES = ["Automática", "640x480", "1280x720", "1920x1080"]


def backend():
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW                                  # abre mucho más rápido que MSMF
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)


def device_names():
    """Nombres de las cámaras cuando el sistema los entrega fácilmente (Linux; Windows con pygrabber)."""
    names = {}
    try:
        if sys.platform.startswith("linux"):
            for p in glob.glob("/sys/class/video4linux/video*/name"):
                idx = int(p.split("video")[-1].split("/")[0])
                with open(p) as f:
                    names[idx] = f.read().strip()
        elif sys.platform.startswith("win"):
            from pygrabber.dshow_graph import FilterGraph          # opcional: pip install pygrabber
            for i, n in enumerate(FilterGraph().get_input_devices()):
                names[i] = n
    except Exception:
        pass
    return names


def candidate_indices(max_index=8):
    if sys.platform.startswith("linux"):
        found = sorted(int(p.replace("/dev/video", "")) for p in glob.glob("/dev/video*") if p.replace("/dev/video", "").isdigit())
        return found or []
    return list(range(max_index))


def scan_cameras(max_index=8, progress=None):
    """Devuelve [{'index', 'name', 'size'}] de las cámaras que entregan imagen."""
    names = device_names()
    idxs = candidate_indices(max_index)
    found = []
    for n, i in enumerate(idxs):
        cap = cv2.VideoCapture(i, backend())
        try:
            if cap.isOpened():
                ok, fr = cap.read()
                if ok and fr is not None:
                    h, w = fr.shape[:2]
                    found.append({"index": i, "name": names.get(i, f"Cámara {i}"), "size": (w, h)})
        finally:
            cap.release()
        if progress:
            progress(n + 1, len(idxs))
    return found


def grab_snapshot(index, width=0, height=0):
    """Una sola imagen de la cámara (para la vista previa del selector)."""
    cap = open_capture(index, width, height, 0)
    if cap is None:
        return None
    try:
        for _ in range(4):                                      # las primeras imágenes suelen salir oscuras
            ok, fr = cap.read()
        return fr if ok else None
    finally:
        cap.release()


def open_capture(index, width=0, height=0, fps=0):
    cap = cv2.VideoCapture(int(index), backend())
    if not cap.isOpened():
        cap.release()
        return None
    if width and height:
        if width >= 1280:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # más fotogramas por segundo en USB
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraGrabber:
    """Lee la cámara en un hilo y guarda solo el último fotograma: así no se acumula retraso."""

    def __init__(self, cap, mirror=True):
        self.cap, self.mirror = cap, mirror
        self.alive = True
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._seen = 0
        self._fails = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.alive:
            ok, fr = self.cap.read()
            if not ok or fr is None:
                self._fails += 1
                if self._fails > 250:
                    self.alive = False
                time.sleep(0.01)
                continue
            self._fails = 0
            if self.mirror:
                fr = cv2.flip(fr, 1)
            with self._lock:
                self._frame, self._seq = fr, self._seq + 1

    def latest(self):
        """Devuelve el fotograma nuevo (si lo hay desde la última vez) o None."""
        with self._lock:
            if self._seq == self._seen:
                return None
            self._seen = self._seq
            return self._frame

    def stop(self):
        self.alive = False
        self.thread.join(timeout=1.0)
        self.cap.release()
