"""Ventana de salida para proyectar: solo la imagen, sin controles, en pantalla completa y en otro monitor."""
from __future__ import annotations

import tkinter as tk

import cv2
from PIL import Image, ImageTk

try:
    from screeninfo import get_monitors          # opcional: pip install screeninfo
except Exception:
    get_monitors = None

FIT_MODES = ["ajustar", "llenar", "estirar"]


def list_monitors():
    """[{'name', 'x', 'y', 'w', 'h'}] de los monitores conectados (si screeninfo está instalado)."""
    out = []
    if get_monitors is not None:
        try:
            for i, m in enumerate(get_monitors()):
                out.append({"name": f"Monitor {i + 1}: {m.width}x{m.height}" + (" (principal)" if getattr(m, "is_primary", False) else ""),
                            "x": m.x, "y": m.y, "w": m.width, "h": m.height})
        except Exception:
            pass
    return out


class ProjectorWindow(tk.Toplevel):
    def __init__(self, master, on_close=None, monitor=None, mode="ajustar"):
        super().__init__(master)
        self.on_close, self.monitor, self.mode = on_close, monitor, mode
        self.title("Salida — proyección")
        self.configure(bg="black")
        self.geometry("960x540+140+120")
        self.cv = tk.Canvas(self, bg="black", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.black, self.fullscreen, self.last, self.photo, self.item = False, False, None, None, None
        self._saved_geo, self._hide_job, self._hint_job = None, None, None
        for k in ("<F11>", "<f>", "<F>"):
            self.bind(k, lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.toggle_fullscreen(False))
        self.bind("<Double-Button-1>", lambda e: self.toggle_fullscreen())
        for k in ("<b>", "<B>"):
            self.bind(k, lambda e: self.toggle_black())
        for k in ("<m>", "<M>"):
            self.bind(k, lambda e: self.cycle_mode())
        self.bind("<Motion>", self._on_motion)
        self.cv.bind("<Configure>", lambda e: self.redraw())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.focus_force()

    # -- imagen ---------------------------------------------------------------
    def show(self, bgr):
        self.last = bgr
        self.redraw()

    def redraw(self):
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        if cw < 8 or ch < 8:
            return
        if self.black or self.last is None:
            self.cv.delete("img")
            self.item = None
            return
        h, w = self.last.shape[:2]
        if self.mode == "estirar":
            nw, nh = cw, ch
        elif self.mode == "llenar":
            s = max(cw / w, ch / h)
            nw, nh = int(w * s) + 1, int(h * s) + 1
        else:
            s = min(cw / w, ch / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
        img = cv2.resize(self.last, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if self.mode == "llenar":
            x0, y0 = (nw - cw) // 2, (nh - ch) // 2
            img = img[y0:y0 + ch, x0:x0 + cw]
            nw, nh = img.shape[1], img.shape[0]
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        if self.item is None:
            self.item = self.cv.create_image(ox, oy, anchor="nw", image=self.photo, tags="img")
        else:
            self.cv.coords(self.item, ox, oy)
            self.cv.itemconfig(self.item, image=self.photo)

    # -- controles --------------------------------------------------------------
    def toggle_black(self):
        self.black = not self.black
        self.redraw()
        self._hint("PANTALLA EN NEGRO" if self.black else "")

    def cycle_mode(self):
        self.mode = FIT_MODES[(FIT_MODES.index(self.mode) + 1) % len(FIT_MODES)]
        self.redraw()
        self._hint(f"Ajuste: {self.mode}")

    def set_mode(self, mode):
        self.mode = mode
        self.redraw()

    def set_monitor(self, mon):
        self.monitor = mon
        if self.fullscreen:
            self.toggle_fullscreen(False)
            self.toggle_fullscreen(True)

    def toggle_fullscreen(self, on=None):
        on = (not self.fullscreen) if on is None else on
        if on == self.fullscreen:
            return
        if on:
            self._saved_geo = self.geometry()
            if self.monitor:                                   # cubre exactamente el monitor elegido
                m = self.monitor
                self.overrideredirect(True)
                self.geometry(f"{m['w']}x{m['h']}+{m['x']}+{m['y']}")
            else:
                self.attributes("-fullscreen", True)
            self.attributes("-topmost", True)
            self.focus_force()
            self._hint("Esc: salir · B: negro · M: ajuste")
            self._schedule_cursor_hide()
        else:
            self.attributes("-topmost", False)
            if self.monitor:
                self.overrideredirect(False)
            self.attributes("-fullscreen", False)
            if self._saved_geo:
                self.geometry(self._saved_geo)
            self.cv.configure(cursor="arrow")
        self.fullscreen = on
        self.after(80, self.redraw)

    def _hint(self, text):
        self.cv.delete("hint")
        if self._hint_job:
            self.after_cancel(self._hint_job)
        if text:
            self.cv.create_text(16, 14, text=text, anchor="nw", fill="#9a9aa6", font=("TkDefaultFont", 12), tags="hint")
            self._hint_job = self.after(2500, lambda: self.cv.delete("hint"))

    def _on_motion(self, _e=None):
        self.cv.configure(cursor="arrow")
        self._schedule_cursor_hide()

    def _schedule_cursor_hide(self):
        if self._hide_job:
            self.after_cancel(self._hide_job)
        if self.fullscreen:
            self._hide_job = self.after(1800, lambda: self.cv.configure(cursor="none"))

    def close(self):
        try:
            if self._hide_job:
                self.after_cancel(self._hide_job)
        except Exception:
            pass
        self.destroy()
        if self.on_close:
            self.on_close()
