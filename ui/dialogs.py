"""Diálogos: selector de cámara (varias entradas) y herramienta de datamosh real."""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from core import cameras
from core import datamosh_real as dm
from ui import theme as T
from ui.widgets import FlatButton, Toggle, section


class BackgroundMixin:
    """Ejecuta trabajo en un hilo y entrega el resultado en el hilo de Tk (Tk no es seguro entre hilos)."""

    def _bg_init(self):
        self._bgq = queue.Queue()
        self._bg_poll()

    def _bg_poll(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                fn = self._bgq.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.after(80, self._bg_poll)

    def run_bg(self, work, done):
        def runner():
            try:
                res, err = work(), None
            except Exception as e:           # noqa: BLE001
                res, err = None, str(e)
            self._bgq.put(lambda: done(res, err))
        threading.Thread(target=runner, daemon=True).start()


def _row(parent, label, widget_factory):
    f = tk.Frame(parent, bg=T.PANEL)
    f.pack(fill="x", pady=4)
    tk.Label(f, text=label, font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, width=28, anchor="w").pack(side="left")
    w = widget_factory(f)
    w.pack(side="left", fill="x", expand=True)
    return w


class CameraDialog(BackgroundMixin, tk.Toplevel):
    """Lista las cámaras conectadas, muestra una vista previa de la elegida y abre la seleccionada."""

    def __init__(self, master, settings, on_open, busy_index=None, busy_frame_getter=None):
        super().__init__(master)
        self._bg_init()
        self.on_open, self.settings = on_open, settings
        self.busy_index, self.busy_frame_getter = busy_index, busy_frame_getter
        self.title("Entrada de cámara")
        self.configure(bg=T.PANEL, padx=18, pady=16)
        self.transient(master)
        self.resizable(False, False)
        self.cams, self._token = [], 0
        tk.Label(self, text="Entrada de cámara", font=T.FONT_TITLE, fg=T.TEXT, bg=T.PANEL).pack(anchor="w")
        tk.Label(self, text="Elige qué cámara usar (capturadoras, webcams, cámaras USB…).", font=T.FONT_SMALL, fg=T.MUTED,
                 bg=T.PANEL).pack(anchor="w", pady=(2, 10))
        body = tk.Frame(self, bg=T.PANEL)
        body.pack(fill="x")
        left = tk.Frame(body, bg=T.PANEL)
        left.pack(side="left", fill="y")
        self.lb = tk.Listbox(left, width=44, height=9, bg=T.CARD, fg=T.TEXT, selectbackground=T.ACCENT, activestyle="none",
                             highlightthickness=0, borderwidth=0, font=T.FONT, exportselection=False)
        self.lb.pack()
        self.status = tk.Label(left, text="Buscando cámaras…", font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, anchor="w")
        self.status.pack(fill="x", pady=(6, 0))
        right = tk.Frame(body, bg=T.PANEL)
        right.pack(side="left", padx=(16, 0))
        self.prev = tk.Label(right, text="Vista previa", bg=T.CARD, fg=T.FAINT, width=40, height=11)
        self.prev.pack()
        self._img = None
        opts = tk.Frame(self, bg=T.PANEL)
        opts.pack(fill="x", pady=(14, 0))
        self.res = tk.StringVar(value=settings.get("cam_res", cameras.RESOLUCIONES[0]))
        self.fps = tk.StringVar(value=str(settings.get("cam_fps", "Automático")))
        _row(opts, "Resolución", lambda p: ttk.Combobox(p, textvariable=self.res, values=cameras.RESOLUCIONES, state="readonly"))
        _row(opts, "Fotogramas por segundo", lambda p: ttk.Combobox(p, textvariable=self.fps, values=["Automático", "30", "60"], state="readonly"))
        mrow = tk.Frame(opts, bg=T.PANEL)
        mrow.pack(fill="x", pady=4)
        tk.Label(mrow, text="Espejo (como un selfie)", font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, width=28, anchor="w").pack(side="left")
        self.mirror = Toggle(mrow, settings.get("cam_mirror", True))
        self.mirror.pack(side="left")
        btns = tk.Frame(self, bg=T.PANEL)
        btns.pack(fill="x", pady=(16, 0))
        FlatButton(btns, "Buscar de nuevo", self.scan, "solid").pack(side="left")
        FlatButton(btns, "Cancelar", self.destroy, "ghost").pack(side="right")
        FlatButton(btns, "Abrir cámara", self._open, "primary").pack(side="right", padx=(0, 8))
        self.lb.bind("<<ListboxSelect>>", lambda e: self._preview())
        self.lb.bind("<Double-Button-1>", lambda e: self._open())
        self.scan()

    def _size(self):
        r = self.res.get()
        if "x" in r:
            w, h = r.split("x")
            return int(w), int(h)
        return 0, 0

    def scan(self):
        self.lb.delete(0, "end")
        self.status.configure(text="Buscando cámaras…")
        self._token += 1
        token = self._token

        self.run_bg(lambda: cameras.scan_cameras(8), lambda res, err: self._scanned(res or [], token))

    def _scanned(self, found, token):
        if token != self._token or not self.winfo_exists():
            return
        self.cams = found
        for c in found:
            self.lb.insert("end", f"  {c['name']}   ·   {c['size'][0]}x{c['size'][1]}")
        if not found:
            self.status.configure(text="No se encontró ninguna cámara. Conéctala y pulsa «Buscar de nuevo».")
            return
        self.status.configure(text=f"{len(found)} cámara(s) encontrada(s)")
        last = self.settings.get("cam_index")
        sel = next((i for i, c in enumerate(found) if c["index"] == last), 0)
        self.lb.selection_set(sel)
        self._preview()

    def _preview(self):
        cur = self.lb.curselection()
        if not cur:
            return
        cam = self.cams[cur[0]]
        token = self._token = self._token + 1

        size = self._size()

        def work():
            if cam["index"] == self.busy_index and self.busy_frame_getter:
                return self.busy_frame_getter()                    # la cámara ya está en uso: se muestra su última imagen
            return cameras.grab_snapshot(cam["index"], *size)

        self.run_bg(work, lambda fr, err: self._show(fr, token))

    def _show(self, fr, token):
        if token != self._token or not self.winfo_exists():
            return
        if fr is None:
            self.prev.configure(image="", text="Sin imagen (¿otra aplicación la está usando?)")
            return
        h, w = fr.shape[:2]
        s = min(320 / w, 200 / h)
        im = cv2.resize(fr, (int(w * s), int(h * s)))
        self._img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        self.prev.configure(image=self._img, text="", width=im.shape[1], height=im.shape[0])

    def _open(self):
        cur = self.lb.curselection()
        if not cur:
            return
        cam = self.cams[cur[0]]
        w, h = self._size()
        fps = 0 if self.fps.get() == "Automático" else int(self.fps.get())
        self.settings.update({"cam_index": cam["index"], "cam_res": self.res.get(), "cam_fps": self.fps.get(),
                              "cam_mirror": self.mirror.value})
        self.destroy()
        self.on_open({"index": cam["index"], "name": cam["name"], "width": w, "height": h, "fps": fps,
                      "mirror": self.mirror.value})


class DatamoshDialog(BackgroundMixin, tk.Toplevel):
    """Datamosh REAL sobre archivos (manipula el códec): clásico, bloom, combinar y barajar."""

    HELP = {
        dm.MODES[0]: "Quita los fotogramas clave (I-frames): el movimiento del video se aplica sobre la imagen anterior y todo se derrite.",
        dm.MODES[1]: "Repite varias veces los mismos fotogramas P: el movimiento se acumula y las formas «florecen».",
        dm.MODES[2]: "Une varios videos y quita el I-frame de cada transición: el movimiento del siguiente arrastra la imagen del anterior.",
        dm.MODES[3]: "Corta el video en trozos y los baraja; cada trozo aplica su movimiento sobre el anterior.",
    }

    def __init__(self, master, initial_file=None):
        super().__init__(master)
        self._bg_init()
        self.title("Datamosh real (códec)")
        self.configure(bg=T.PANEL, padx=18, pady=16)
        self.transient(master)
        self.resizable(False, False)
        tk.Label(self, text="Datamosh real", font=T.FONT_TITLE, fg=T.TEXT, bg=T.PANEL).pack(anchor="w")
        tk.Label(self, text="Rompe de verdad la compresión del video (no es una simulación). Se guarda como un MP4 nuevo.",
                 font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, wraplength=520, justify="left").pack(anchor="w", pady=(2, 10))
        self.mode = tk.StringVar(value=dm.MODES[0])
        _row(self, "Técnica", lambda p: ttk.Combobox(p, textvariable=self.mode, values=dm.MODES, state="readonly"))
        self.help = tk.Label(self, text=self.HELP[dm.MODES[0]], font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, wraplength=520, justify="left")
        self.help.pack(anchor="w", pady=(0, 8))
        self.mode.trace_add("write", lambda *_: self.help.configure(text=self.HELP.get(self.mode.get(), "")))

        section(self, "Videos").pack(fill="x", pady=(4, 6))
        fr = tk.Frame(self, bg=T.PANEL)
        fr.pack(fill="x")
        self.files = tk.Listbox(fr, height=3, bg=T.CARD, fg=T.TEXT, highlightthickness=0, borderwidth=0, font=T.FONT_SMALL,
                                selectbackground=T.ACCENT, activestyle="none")
        self.files.pack(side="left", fill="x", expand=True)
        col = tk.Frame(fr, bg=T.PANEL)
        col.pack(side="left", padx=(8, 0))
        FlatButton(col, "Añadir…", self._add, "solid").pack(fill="x")
        FlatButton(col, "Quitar", lambda: self.files.delete(self.files.curselection()[0]) if self.files.curselection() else None, "ghost").pack(fill="x", pady=(4, 0))
        if initial_file:
            self.files.insert("end", initial_file)

        section(self, "Parámetros").pack(fill="x", pady=(12, 4))
        self.gop = tk.StringVar(value="2.0")
        self.remove = tk.StringVar(value="todos")
        self.rng_a, self.rng_b = tk.StringVar(value="0"), tk.StringVar(value="9999")
        self.times = tk.StringVar(value="2.0")
        self.count = tk.StringVar(value="30")
        self.length = tk.StringVar(value="2")
        self.quality = tk.StringVar(value="4")
        self.width = tk.StringVar(value="Original")
        _row(self, "I-frame cada (s)", lambda p: ttk.Spinbox(p, from_=0.3, to=20, increment=0.1, textvariable=self.gop, width=8))
        _row(self, "I-frames a quitar", lambda p: ttk.Combobox(p, textvariable=self.remove, values=["todos", "uno de cada 2", "uno de cada 3"], state="readonly", width=16))
        _row(self, "Solo entre (s)", lambda p: self._pair(p))
        _row(self, "Bloom: segundos de inicio", lambda p: ttk.Entry(p, textvariable=self.times))
        _row(self, "Bloom: repeticiones", lambda p: ttk.Spinbox(p, from_=1, to=300, textvariable=self.count, width=8))
        _row(self, "Bloom: fotogramas P a repetir", lambda p: ttk.Spinbox(p, from_=1, to=20, textvariable=self.length, width=8))
        _row(self, "Calidad (2 = alta, 12 = baja)", lambda p: ttk.Spinbox(p, from_=2, to=12, textvariable=self.quality, width=8))
        _row(self, "Ancho de salida", lambda p: ttk.Combobox(p, textvariable=self.width, values=["Original", "1280", "960", "640"], state="readonly", width=12))
        arow = tk.Frame(self, bg=T.PANEL)
        arow.pack(fill="x", pady=4)
        tk.Label(arow, text="Conservar el audio original", font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, width=28, anchor="w").pack(side="left")
        self.audio = Toggle(arow, True)
        self.audio.pack(side="left")

        self.msg = tk.Label(self, text="", font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, wraplength=520, justify="left")
        self.msg.pack(anchor="w", pady=(10, 0))
        btns = tk.Frame(self, bg=T.PANEL)
        btns.pack(fill="x", pady=(10, 0))
        FlatButton(btns, "Cerrar", self.destroy, "ghost").pack(side="right")
        self.run_btn = FlatButton(btns, "Crear video con datamosh", self._run, "primary")
        self.run_btn.pack(side="right", padx=(0, 8))

    def _pair(self, p):
        f = tk.Frame(p, bg=T.PANEL)
        ttk.Entry(f, textvariable=self.rng_a, width=6).pack(side="left")
        tk.Label(f, text="a", fg=T.MUTED, bg=T.PANEL).pack(side="left", padx=6)
        ttk.Entry(f, textvariable=self.rng_b, width=6).pack(side="left")
        return f

    def _add(self):
        for p in filedialog.askopenfilenames(parent=self, filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"), ("Todos", "*.*")]):
            self.files.insert("end", p)

    def _run(self):
        srcs = list(self.files.get(0, "end"))
        if not srcs:
            messagebox.showinfo("Datamosh real", "Añade al menos un video.", parent=self)
            return
        mode = self.mode.get()
        if mode == dm.MODES[2] and len(srcs) < 2:
            messagebox.showinfo("Datamosh real", "Para combinar necesitas al menos dos videos.", parent=self)
            return
        out = filedialog.asksaveasfilename(parent=self, defaultextension=".mp4", filetypes=[("MP4", "*.mp4")], initialfile="datamosh.mp4")
        if not out:
            return
        try:
            times = [float(t) for t in self.times.get().replace(";", ",").split(",") if t.strip()]
            kw = dict(sources=srcs, out_path=out, mode=mode, gop_seconds=float(self.gop.get()),
                      remove={"todos": "todos"}.get(self.remove.get(), "cada"),
                      every={"todos": 1, "uno de cada 2": 2, "uno de cada 3": 3}[self.remove.get()],
                      range_s=(float(self.rng_a.get()), float(self.rng_b.get())), repeat_times=times,
                      repeat_count=int(self.count.get()), repeat_len=int(self.length.get()), quality=int(float(self.quality.get())),
                      keep_audio=self.audio.value)
            if self.width.get() != "Original":
                fps, n, w0, h0 = dm.video_info(srcs[0])
                wd = int(self.width.get())
                kw["size"] = (wd, int(h0 * wd / w0))
        except ValueError:
            messagebox.showerror("Datamosh real", "Revisa los números de los parámetros.", parent=self)
            return
        self.run_btn.set_enabled(False)
        self.msg.configure(text="Trabajando… (puede tardar un poco)", fg=T.WARN)

        def work():
            return dm.datamosh_file(log=lambda m: self._bgq.put(lambda: self.msg.configure(text=m)), **kw)

        self.run_bg(work, lambda res, err: self._done(res, err))

    def _done(self, res, err):
        if not self.winfo_exists():
            return
        self.run_btn.set_enabled(True)
        if err:
            self.msg.configure(text=err, fg=T.DANGER)
        else:
            self.msg.configure(text=res, fg=T.GOOD)
