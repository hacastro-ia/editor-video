"""Fuente de video: archivo o cámara (con captura en un hilo)."""
from __future__ import annotations

import cv2

from core import cameras


class VideoSource:
    def __init__(self):
        self.cap = None
        self.grabber = None
        self.path = None
        self.is_camera = False
        self.camera_index = None
        self.fps = 30.0
        self.total = 0

    @property
    def is_open(self):
        if self.is_camera:
            return self.grabber is not None and self.grabber.alive
        return self.cap is not None and self.cap.isOpened()

    def close(self):
        if self.grabber is not None:
            self.grabber.stop()
            self.grabber = None
        elif self.cap is not None:
            self.cap.release()
        self.cap = None
        self.path = None
        self.is_camera = False
        self.camera_index = None

    def open_file(self, path):
        self.close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self.cap, self.path, self.is_camera = cap, path, False
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps < 1 or self.fps > 240:
            self.fps = 30.0
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def open_camera(self, index=0, width=0, height=0, fps=0, mirror=True, cap=None):
        """Abre una cámara por índice (o usa un `cap` ya abierto, útil para pruebas)."""
        self.close()
        cap = cap if cap is not None else cameras.open_capture(index, width, height, fps)
        if cap is None or not cap.isOpened():
            return False
        self.cap, self.is_camera, self.path, self.camera_index = cap, True, None, index
        real = cap.get(cv2.CAP_PROP_FPS)
        self.fps = real if 5 <= (real or 0) <= 240 else 30.0
        self.total = 0
        self.grabber = cameras.CameraGrabber(cap, mirror)
        return True

    def read_camera(self):
        """('frame', imagen) si hay un fotograma nuevo, ('wait', None) si todavía no, ('dead', None) si se perdió."""
        if self.grabber is None or not self.grabber.alive:
            return "dead", None
        fr = self.grabber.latest()
        return ("frame", fr) if fr is not None else ("wait", None)

    def frame_size(self):
        if self.cap is None:
            return 0, 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        if self.is_camera or not self.is_open:
            return False, None
        return self.cap.read()

    def grab(self):
        return self.cap.grab() if self.is_open and not self.is_camera else False

    def seek(self, index):
        if self.is_open and not self.is_camera:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
