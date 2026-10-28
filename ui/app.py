"""Editor de video — lógica de la aplicación: reproducción, cámara, proyección, Auto-VJ y exportación."""
from __future__ import annotations

import os
import queue
import random
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import effects  # noqa: F401  (registra todos los efectos)
from core import settings
from core.audio import AudioAnalysis, LiveAudio
from core.exporter import export_video
from core.facecore import FaceState
from core.library import open_folder
from core.params import Context, zero_audio
from core.pipeline import REGISTRY, Chain
from core.presets import PRESETS, build_preset
from core.source import VideoSource
from core.transport import FrameFetcher, build_plan
from core.utils import prepare_frame
from ui import theme as T
from ui.dialogs import CameraDialog, DatamoshDialog
from ui.panels import PanelsMixin
from ui.projector import ProjectorWindow, list_monitors
from ui.widgets import FlatButton

VJ_SKIP = ("Personas", "Biblioteca", "Detección", "Privacidad", "Traqueo")      # presets que necesitan modelos pesados


def _fmt_time(sec):
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


class App(PanelsMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Editor de video")
        self.geometry("1420x860")
        self.minsize(1180, 720)
        self.settings = settings.load()

        self.source = VideoSource()
        self.chain = Chain()
        self.faces = FaceState()
        self.ctx = Context()
        self.ctx.faces = self.faces
        self.analysis = None
        self.live = LiveAudio()
        self.live_on = False

        self.playing = False
        self.transport = "pause"          # play | pause | rewind | ff
        self.direction, self.speed = 1, 1
        self.plan_mode, self.plan, self.fetcher, self.plan_t0, self._plan_j = False, None, None, 0.0, -1
        self.events = []                  # guion de cinta
        self._last_live = 0.0
        self.busy = False
        self.cur_idx = 0
        self.play_t0 = 0.0
        self.play_i0 = 0
        self.last_frame_raw = None
        self.last_out = None
        self.photo = None
        self.img_item = None
        self.disp_geom = None
        self.dragging = False
        self._refresh_pending = False
        self._queue = queue.Queue()
        self._last_meter = 0.0
        self._meter_count = 0
        self._faces_sig = None
        self._face_vars, self._face_imgs, self._face_toggles = {}, [], {}
        self._last_frame_t = None
        self._fps_avg = 0.0
        self.selected_effect = None
        self.current_tab = "Efectos"
        self.projector = None
        self.monitors = []
        self._vj_t0, self._vj_beats, self._vj_prev = time.perf_counter(), 0, 0.0

        self._build_ui()
        self.monitors = list_monitors()
        self.monitor_cb.configure(values=["Pantalla actual"] + [m["name"] for m in self.monitors])
        self._refresh_list()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._tick)

    # ------------------------------------------------------------------ atajos de teclado
    def _bind_keys(self):
        def guard(fn):
            def handler(e):
                if isinstance(self.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Text)):
                    return None
                fn()
                return "break"
            return handler

        keys = {"<space>": self.toggle_play, "<j>": lambda: self.toggle_search(-1), "<l>": lambda: self.toggle_search(1),
                "<k>": self._pause, "<p>": self.toggle_projector, "<b>": self._toggle_blackout, "<r>": self._randomize_selected,
                "<Control-e>": self.export, "<Control-o>": self.open_video}
        for k, fn in keys.items():
            self.bind(k, guard(fn))

    def _toggle_blackout(self):
        if self.projector is not None:
            self.projector.toggle_black()

    def _randomize_selected(self):
        if self.selected_effect is not None:
            self._randomize_effect(self.selected_effect)

    # ------------------------------------------------------------------ cámara
    def _set_audio_src(self, v):
        self.audio_src.set(v)
        self._audio_pills.select(v)

    def open_camera_dialog(self):
        if self.busy:
            return
        CameraDialog(self, self.settings, self._open_camera,
                     busy_index=self.source.camera_index if self.source.is_camera else None,
                     busy_frame_getter=lambda: self.last_frame_raw)

    def _open_camera(self, cfg):
        self._pause()
        self._stop_live()
        if not self.source.open_camera(cfg["index"], cfg["width"], cfg["height"], cfg["fps"], cfg["mirror"]):
            messagebox.showerror("Cámara", f"No se pudo abrir «{cfg['name']}». ¿La está usando otra aplicación?")
            return
        self.analysis = None
        self._reset_state()
        self.cur_idx = 0
        self.seek.state(["disabled"])
        self.seek.set_markers([])
        self._set_audio_src("Micrófono")
        self._on_audio_src()
        w, h = self.source.frame_size()
        self.cam_label.set(f"{cfg['name']}  ·  {w}x{h}")
        self._play()
        self.status(f"Cámara «{cfg['name']}» activa ({w}x{h}). Para exportar necesitas un video de archivo.")
        settings.save(self.settings)

    # ------------------------------------------------------------------ proyector
    def _selected_monitor(self):
        name = self.monitor_var.get()
        return next((m for m in self.monitors if m["name"] == name), None)

    def toggle_projector(self):
        if self.projector is not None:
            self.projector.close()
            return
        self.projector = ProjectorWindow(self, on_close=self._on_projector_closed, monitor=self._selected_monitor(),
                                         mode=self.fit_pills.value)
        self.btn_project.set_active(True)
        self.btn_project.configure(text="Proyectando")
        self.btn_proj2.configure(text="Cerrar ventana de proyección")
        self.proj_var.set("● proyectando")
        if self.last_out is not None:
            self.projector.show(self.last_out)
        self.status("Ventana de proyección abierta. Arrástrala al proyector y pulsa F para pantalla completa.")

    def _on_projector_closed(self):
        self.projector = None
        self.btn_project.set_active(False)
        self.btn_project.configure(text="Proyectar")
        self.btn_proj2.configure(text="Abrir ventana de proyección")
        self.proj_var.set("")
        self._redraw()

    def projector_fullscreen(self):
        if self.projector is None:
            self.toggle_projector()
        self.projector.set_monitor(self._selected_monitor())
        self.projector.toggle_fullscreen(True)

    def _on_monitor(self):
        if self.projector is not None:
            self.projector.set_monitor(self._selected_monitor())

    def _on_fit(self, v):
        if self.projector is not None:
            self.projector.set_mode(v)

    # ------------------------------------------------------------------ Auto-VJ
    def _on_vj(self, v):
        self.vj_on.set(v)
        self._vj_t0, self._vj_beats = time.perf_counter(), 0

    def _vj_pool(self):
        return [n for n in PRESETS if not any(k in n for k in VJ_SKIP)]

    def vj_next(self):
        pool = self._vj_pool()
        cur = getattr(self, "_vj_last", None)
        name = random.choice([n for n in pool if n != cur] or pool)
        self._vj_last = name
        self._vj_t0, self._vj_beats = time.perf_counter(), 0
        self.apply_preset(name, keep_pinned=True)

    def _vj_check(self, beat):
        hit = beat > 0.95 and self._vj_prev <= 0.95
        self._vj_prev = beat
        if not self.vj_on.get():
            return
        if self.vj_beat.get():
            self._vj_beats += int(hit)
            if self._vj_beats >= 16:
                self.vj_next()
            return
        try:
            secs = max(2.0, float(self.vj_secs.get()))
        except ValueError:
            secs = 12.0
        if time.perf_counter() - self._vj_t0 > secs:
            self.vj_next()

    def apply_preset(self, name, keep_pinned=False):
        pinned = [e for e in self.chain.effects if e.pin_first] if keep_pinned else []
        self.chain.clear()
        for e in pinned + build_preset(name):
            self.chain.add(e)
        self._refresh_list(select=len(pinned) if len(self.chain.effects) > len(pinned) else (0 if self.chain.effects else None))
        self._reset_state()
        self._request_refresh()
        self.status(f"Preset «{name}» aplicado.")

    def open_datamosh_tool(self):
        DatamoshDialog(self, initial_file=self.source.path if (self.source.is_open and not self.source.is_camera) else None)

    def _set_transport(self, t):
        self.transport = t
        self.playing = t in ("play", "rewind", "ff")
        self.btn_play.configure(text="⏸" if t == "play" else "▶")
        self.btn_rew.set_active(t == "rewind")
        self.btn_ff.set_active(t == "ff")

    def _on_close(self):
        try:
            self._pause()
            self._stop_live()
            self.source.close()
            if self.projector is not None:
                self.projector.close()
            settings.save(self.settings)
        finally:
            self.destroy()

    def add_effect(self, cls):
        e = cls()
        idx = self.chain.add(e)
        self._refresh_list(select=idx)
        self._request_refresh()

    def open_library(self):
        folder = ""
        for e in self.chain.effects:
            if "carpeta" in e.params and e.params["carpeta"].value.strip():
                folder = e.params["carpeta"].value.strip()
        p = open_folder(folder or None)
        self.status(f"Biblioteca de imágenes: {p}  (una subcarpeta por clase: person, dog, car…)")

    def _max_width(self):
        v = self.res_var.get()
        return 0 if v == "Original" else int(v)

    def status(self, text):
        self.status_var.set(text)

    def _post(self, fn):
        self._queue.put(fn)

    def _reset_state(self):
        self.chain.reset()
        self.faces.clear_tracks()

    def open_video(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Elegir video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.wmv *.mpg *.mpeg"), ("Todos", "*.*")])
        if not path:
            return
        self._pause()
        self._stop_live()
        if not self.source.open_file(path):
            messagebox.showerror("Abrir video", "No se pudo abrir ese archivo.")
            return
        self.analysis = None
        self.seek.state(["!disabled"])
        self.seek.configure(from_=0, to=max(1, self.source.total - 1))
        self._set_audio_src("Del video")
        self._load_audio_async(path)
        self._reset_state()
        self.clear_events()
        self.cam_label.set("Sin cámara activa")
        self.cur_idx = 0
        self.seek_to(0)
        w = int(self.source.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status(f"{os.path.basename(path)} — {w}x{h}, {self.source.fps:.1f} fps, {self.source.total} fotogramas")

    def _load_audio_async(self, path):
        self.audio_note.set("Analizando el audio del video…")

        def work():
            a, err = AudioAnalysis(), ""
            try:
                a.load(path)
            except Exception as e:  # sin pista de audio, sin FFmpeg, etc.
                a, err = None, str(e)
            self._post(lambda: self._audio_ready(path, a, err))

        threading.Thread(target=work, daemon=True).start()

    def _audio_ready(self, path, a, err):
        if path != self.source.path:
            return
        if a is None:
            self.analysis = None
            self.audio_note.set(f"Sin audio utilizable: {err}")
            return
        a.set_sensitivity(self.sens_var.get())
        a.muted = self.mute_var.get()
        self.analysis = a
        self.audio_note.set(f"Audio analizado ({a.duration:.1f} s).")
        if self.transport == "play" and self.audio_src.get() == "Del video":
            self._start_audio(self.cur_idx / self.source.fps)

    def _stop_live(self):
        if self.live_on:
            self.live.stop()
            self.live_on = False

    def _on_audio_src(self, _e=None):
        src = self.audio_src.get()
        if self.analysis:
            self.analysis.stop()
        if src == "Micrófono":
            if not self.live_on:
                self.live_on = self.live.start()
                if not self.live_on:
                    messagebox.showwarning("Micrófono", self.live.error)
                    self._set_audio_src("Ninguno")
                else:
                    self.live.set_sensitivity(self.sens_var.get())
        else:
            self._stop_live()
        if src == "Del video" and self.transport == "play" and self.source.path:
            self._start_audio(self.cur_idx / self.source.fps)

    def _on_mute(self):
        if self.analysis:
            self.analysis.muted = self.mute_var.get()
            if self.mute_var.get():
                self.analysis.stop()
            elif self.transport == "play" and self.audio_src.get() == "Del video":
                self._start_audio(self.cur_idx / self.source.fps)

    def _on_bpm(self):
        try:
            self.ctx.bpm = min(max(float(self.bpm_var.get()), 20.0), 400.0)
            self.ctx.beats_per_cycle = min(max(float(self.cycle_var.get()), 0.25), 64.0)
        except ValueError:
            pass
        self._request_refresh()

    def _on_sens(self, v):
        s = float(v)
        if self.analysis:
            self.analysis.set_sensitivity(s)
        self.live.set_sensitivity(s)

    def toggle_play(self):
        if self.busy:
            return
        if self.transport == "play":
            self._pause()
        elif self.source.is_open:
            self._play()

    def toggle_search(self, direction):
        """◀◀ / ▶▶: rebobinar o avanzar rápido (como una videocámara). Pulsar otra vez detiene la búsqueda."""
        if self.busy or not self.source.is_open or self.source.is_camera:
            return
        want = "rewind" if direction < 0 else "ff"
        if self.transport == want and not self.plan_mode:
            self._pause()
            self._request_refresh()
            return
        self._start_search(direction)

    def _start_search(self, direction):
        if self.analysis:
            self.analysis.stop()                      # una cinta en búsqueda no suena
        self.plan_mode = False
        self.direction = 1 if direction > 0 else -1
        try:
            self.speed = max(2, int(self.search_speed.get()))
        except ValueError:
            self.speed = 8
        self._set_transport("rewind" if direction < 0 else "ff")
        self.play_t0, self.play_i0 = time.perf_counter(), self.cur_idx
        self._last_frame_t = None

    def _start_audio(self, t):
        if self.analysis and self.audio_src.get() == "Del video" and not self.source.is_camera:
            self.analysis.muted = self.mute_var.get()
            self.analysis.play(t)

    def _play(self):
        if not self.source.is_open:
            return
        self.plan_mode = False
        self.direction, self.speed = 1, 1
        self._set_transport("play")
        self.play_t0 = time.perf_counter()
        self.play_i0 = self.cur_idx
        self._last_frame_t = None
        self._start_audio(self.cur_idx / self.source.fps)

    def _pause(self):
        self.plan_mode = False
        self._set_transport("pause")
        if self.analysis:
            self.analysis.stop()

    def _on_seek_release(self, _e=None):
        self.dragging = False
        if self.source.is_open and not self.source.is_camera and not self.busy:
            self.seek_to(int(float(self.seek.get())))

    def seek_to(self, idx):
        if not self.source.is_open or self.source.is_camera:
            return
        idx = int(min(max(0, idx), max(0, self.source.total - 1)))
        was, was_dir = self.transport, self.direction
        if self.playing:
            self._pause()
        self.source.seek(idx)
        ok, frame = self.source.read()
        if not ok:
            return
        self._reset_state()
        self.cur_idx = idx
        self._process_show(frame, idx)
        if was == "play":
            self._play()
        elif was in ("rewind", "ff"):
            self._start_search(was_dir)

    def _advance(self):
        src = self.source
        if not src.is_open:
            self._pause()
            return
        if src.is_camera:
            st, frame = src.read_camera()
            if st == "dead":
                self._pause()
                self.status("La cámara dejó de responder.")
                return
            if st == "wait":
                return
            self.cur_idx += 1
            self._process_show(frame, self.cur_idx)
            return
        now = time.perf_counter()
        if self.plan_mode:
            self._advance_plan(now)
            return
        if self.speed == 1 and self.direction > 0:            # reproducción normal
            target = self.play_i0 + int((now - self.play_t0) * src.fps)
            if target <= self.cur_idx:
                return
            gap = target - self.cur_idx - 1
            if gap > 45:                                       # muy atrasados: volver a sincronizar
                gap = 0
                self.play_t0, self.play_i0 = now, self.cur_idx
                self._start_audio(self.cur_idx / src.fps)
            for _ in range(gap):
                src.grab()
            ok, frame = src.read()
            if not ok:                                         # fin del video
                self._pause()
                if self.loop_var.get():
                    self.seek_to(0)
                    self._play()
                return
            self.cur_idx = self.cur_idx + gap + 1
            self._process_show(frame, self.cur_idx)
            return
        # búsqueda: rebobinar (◀◀) o avanzar rápido (▶▶)
        last = max(0, src.total - 1)
        target = self.play_i0 + self.direction * int((now - self.play_t0) * src.fps * self.speed)
        target = min(max(target, 0), last)
        at_end = (self.direction < 0 and target <= 0) or (self.direction > 0 and target >= last)
        if target == self.cur_idx:
            if at_end:
                self._pause()
            return
        if (self.direction > 0 and target < self.cur_idx) or (self.direction < 0 and target > self.cur_idx):
            return
        if self.direction > 0 and target - self.cur_idx <= 24:
            for _ in range(target - self.cur_idx - 1):
                src.grab()
        else:
            src.seek(target)
        ok, frame = src.read()
        if not ok:
            self._pause()
            return
        self.cur_idx = target
        self._process_show(frame, target)
        if at_end:
            self._pause()                                      # llegó al extremo de la cinta

    def _hold_seconds(self):
        try:
            return max(0.1, float(self.hold_var.get()))
        except ValueError:
            return 0.6

    def preview_plan(self):
        if not self.source.is_open or self.source.is_camera or self.busy:
            return
        plan = build_plan(self.events, self.source.total, self.source.fps, self._hold_seconds())
        if not plan:
            return
        self._pause()
        self.plan, self.plan_mode = plan, True
        self.fetcher = FrameFetcher(self.source.cap)
        self._reset_state()
        self.plan_t0, self._plan_j = time.perf_counter(), -1
        self._set_transport("play")
        self.status("Previsualizando el guion de cinta (sin sonido). Pulsa Pausar para detener.")

    def _advance_plan(self, now):
        j = int((now - self.plan_t0) * self.source.fps)
        if j >= len(self.plan):
            self._pause()
            self.status("Fin del guion de cinta.")
            return
        if j == self._plan_j:
            return
        self._plan_j = j
        idx, state = self.plan[j]
        frame = self.fetcher.get(idx)
        if frame is None:
            return
        self.cur_idx = idx
        self._process_show(frame, idx, state=state, out_idx=j)

    def _event_text(self, ev):
        fps = self.source.fps or 30.0
        t = ev["frame"] / fps
        pos = f"{int(t // 60):02d}:{t % 60:04.1f}"
        if ev["tipo"] == "pausa":
            return f"{pos}  —  Pausa de {ev['valor']:.1f} s"
        if ev["tipo"] == "rebobinar":
            return f"{pos}  —  Rebobinar {ev['valor']:.1f} s  (×{ev['vel']})"
        return f"{pos}  —  Avance rápido {ev['valor']:.1f} s  (×{ev['vel']})"

    def add_event(self, tipo):
        if not self.source.is_open or self.source.is_camera:
            messagebox.showinfo("Guion de cinta", "Abre un video de archivo y colócate donde quieras insertar el evento.")
            return
        try:
            vel = max(2, int(self.search_speed.get()))
        except ValueError:
            vel = 8
        if tipo == "pausa":
            val = simpledialog.askfloat("Pausa", "Duración de la pausa (segundos):", initialvalue=2.0,
                                        minvalue=0.2, maxvalue=60.0, parent=self)
            vel = 1
        elif tipo == "rebobinar":
            val = simpledialog.askfloat("Rebobinar", f"¿Cuántos segundos de video rebobinar?\n(a ×{vel} se ven en pantalla en ~1/{vel} del tiempo)", initialvalue=3.0,
                                        minvalue=0.5, maxvalue=120.0, parent=self)
        else:
            val = simpledialog.askfloat("Avance rápido", f"¿Cuántos segundos de video saltar?\n(a ×{vel} se ven en pantalla en ~1/{vel} del tiempo)", initialvalue=3.0,
                                        minvalue=0.5, maxvalue=120.0, parent=self)
        if val is None:
            return
        self.events.append({"tipo": tipo, "frame": int(self.cur_idx), "valor": float(val), "vel": int(vel)})
        self._refresh_events()
        self.status("Evento agregado al guion de cinta: se aplicará al exportar.")

    def remove_event(self):
        sel = self.ev_list.curselection()
        if sel:
            del self.events[sel[0]]
            self._refresh_events()

    def clear_events(self):
        self.events = []
        self._refresh_events()

    def save_events(self):
        import json
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Guion de cinta", "*.json")],
                                            initialfile="guion_cinta.json")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"eventos": self.events, "pausa_final": self._hold_seconds()}, f, indent=2)
            self.status(f"Guion guardado en {path}")

    def load_events(self):
        import json
        path = filedialog.askopenfilename(filetypes=[("Guion de cinta", "*.json"), ("Todos", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.events = [{"tipo": e["tipo"], "frame": int(e["frame"]), "valor": float(e["valor"]),
                            "vel": int(e.get("vel", 8))} for e in data["eventos"]]
            self.hold_var.set(str(data.get("pausa_final", 0.6)))
        except Exception as e:
            messagebox.showerror("Guion de cinta", f"No se pudo leer el guion: {e}")
            return
        self._refresh_events()

    def _features(self, t):
        src = self.audio_src.get()
        if src == "Del video" and self.analysis is not None:
            # una cinta pausada o en búsqueda no suena: las señales de audio se apagan
            return self.analysis.features_at(t) if self.ctx.transport == "play" else zero_audio()
        if src == "Micrófono" and self.live_on:
            return self.live.features_at(t)
        return zero_audio()

    def _process_show(self, frame, idx, state=None, out_idx=None):
        self.last_frame_raw = frame
        f = prepare_frame(frame, self._max_width())
        ctx = self.ctx
        ctx.transport = state or (self.transport if not self.source.is_camera else "play")
        ctx.frame_index = idx if out_idx is None else out_idx
        ctx.fps = self.source.fps
        if self.source.is_camera:
            ctx.time = time.perf_counter() - self.play_t0
        elif out_idx is not None:
            ctx.time = out_idx / self.source.fps
        else:
            ctx.time = idx / self.source.fps
        ctx.audio = self._features(ctx.time)
        self._vj_check(float(ctx.audio.get("beat", 0.0)))
        out = self.chain.process(f, ctx)
        self.last_out = out
        self._display(out)
        if not self.source.is_camera:
            if not self.dragging:
                self.seek.set(idx)
            self.time_var.set(f"{_fmt_time(idx / self.source.fps)} / {_fmt_time(self.source.total / self.source.fps)}")
        else:
            self.time_var.set(_fmt_time(ctx.time))
        now = time.perf_counter()
        if self.playing and self._last_frame_t:
            inst = 1.0 / max(now - self._last_frame_t, 1e-3)
            self._fps_avg = inst if self._fps_avg == 0 else self._fps_avg * 0.9 + inst * 0.1
            self.fps_var.set(f"{self._fps_avg:.0f} fps")
        self._last_frame_t = now

    def _request_refresh(self):
        if self.playing or self._refresh_pending or self.last_frame_raw is None or self.busy:
            return
        self._refresh_pending = True
        self.after(40, self._do_refresh)

    def _do_refresh(self):
        self._refresh_pending = False
        if self.last_frame_raw is not None and not self.playing and not self.busy:
            self._process_show(self.last_frame_raw, self.cur_idx)

    def _reprocess_current(self):
        self.chain.reset()
        self._request_refresh()

    def _redraw(self):
        if self.last_out is not None:
            self._display(self.last_out)

    def _draw_overlay(self, img, scale):
        dr = ImageDraw.Draw(img)
        for key, box, censor in self.faces.active():
            x, y, w, h = [float(v) * scale for v in box]
            col = (255, 90, 90) if censor else (90, 255, 140)
            dr.rectangle([x, y, x + w, y + h], outline=col, width=2)
            dr.text((x + 3, max(0, y - 13)), key, fill=col)

    def _display(self, bgr):
        if self.projector is not None:
            self.projector.show(bgr)                      # la ventana de proyección recibe cada fotograma
            if self.save_cpu_var.get():
                return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        disp = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        img = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        if self.overlay_var.get() and any(e.enabled and e.name == "Censura de caras" for e in self.chain.effects):
            self._draw_overlay(img, s)
        self.photo = ImageTk.PhotoImage(img)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.disp_geom = (ox, oy, s)
        if self.img_item is None:
            self.img_item = self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        else:
            self.canvas.coords(self.img_item, ox, oy)
            self.canvas.itemconfig(self.img_item, image=self.photo)

    def _on_canvas_click(self, ev):
        if not self.disp_geom or self.last_out is None:
            return
        ox, oy, s = self.disp_geom
        key = self.faces.toggle_at((ev.x - ox) / s, (ev.y - oy) / s)
        if key:
            self.status(f"«{key}» {'elegida' if key in self.faces.selected else 'no elegida'}.")
            self._faces_sig = None          # fuerza actualizar las casillas
            self._request_refresh()

    def _faces_all(self):
        self.faces.select_all()
        self._request_refresh()

    def _faces_none(self):
        self.faces.select_none()
        self._request_refresh()

    def _clear_refs(self):
        self.faces.clear_references()
        self._faces_sig = None

    def load_reference(self):
        path = filedialog.askopenfilename(title="Foto de la persona (una cara clara y de frente)",
                                          filetypes=[("Imagen", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Todos", "*.*")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Foto de referencia", "No se pudo leer la imagen.")
            return
        name = simpledialog.askstring("Nombre", "¿Cómo se llama esta persona?", parent=self)
        if not name or not name.strip():
            return
        self.status("Cargando el modelo de reconocimiento (la primera vez se descarga)…")
        self.update_idletasks()
        try:
            self.faces.add_reference(name.strip(), img)
        except Exception as e:
            messagebox.showerror("Foto de referencia", str(e))
            return
        self._faces_sig = None
        self.status(f"Referencia «{name.strip()}» agregada.")

    def _set_face_selected(self, key, value):
        with self.faces.lock:
            if value:
                self.faces.selected.add(key)
            else:
                self.faces.selected.discard(key)
        self._request_refresh()

    def save_frame(self):
        if self.last_out is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")], initialfile="captura.png")
        if not path:
            return
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", self.last_out)
        if ok:
            buf.tofile(path)
            self.status(f"Captura guardada en {path}")

    def export(self):
        if self.busy:
            return
        if not self.source.is_open or self.source.is_camera or not self.source.path:
            messagebox.showinfo("Exportar", "Abre un video de archivo para exportar (la webcam no se exporta).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")],
                                            initialfile="editado.mp4")
        if not path:
            return
        self._pause()
        self.busy = True
        cancel = threading.Event()
        ctx = Context()
        ctx.faces = self.faces
        ctx.bpm, ctx.beats_per_cycle = self.ctx.bpm, self.ctx.beats_per_cycle
        audio = self.analysis if self.audio_src.get() != "Ninguno" else None
        max_w = self._max_width()               # se lee aquí: Tk no es seguro desde otro hilo
        src_path = self.source.path
        plan = None
        if self.events:
            plan = build_plan(self.events, self.source.total, self.source.fps, self._hold_seconds())
        pcm = self.analysis
        state = {"i": 0, "total": max(1, len(plan) if plan else self.source.total), "msg": "", "err": "", "done": False}

        dlg = tk.Toplevel(self)
        dlg.title("Exportando…")
        dlg.configure(bg=T.PANEL)
        dlg.transient(self)
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Procesando fotogramas con la cadena de efectos…", padding=12).pack()
        bar = ttk.Progressbar(dlg, length=380, maximum=state["total"])
        bar.pack(padx=14)
        lab = tk.StringVar(value="0%")
        ttk.Label(dlg, textvariable=lab, padding=6).pack()
        FlatButton(dlg, "Cancelar", cancel.set, "solid").pack(pady=(0, 12))
        dlg.protocol("WM_DELETE_WINDOW", cancel.set)

        def progress(i, total):
            state["i"], state["total"] = i, max(1, total)

        def work():
            try:
                state["msg"] = export_video(src_path, path, self.chain, ctx, max_w, audio, progress, cancel,
                                            plan=plan, pcm=pcm)
            except Exception as e:
                traceback.print_exc()
                state["err"] = str(e)
            finally:
                state["done"] = True

        threading.Thread(target=work, daemon=True).start()

        def poll():
            bar.configure(maximum=state["total"], value=state["i"])
            lab.set(f"{100 * state['i'] // state['total']}%   ({state['i']} / {state['total']})")
            if not state["done"]:
                self.after(150, poll)
                return
            dlg.destroy()
            self.busy = False
            self._reset_state()
            self.seek_to(self.cur_idx)
            if state["err"]:
                messagebox.showerror("Exportar", state["err"])
            else:
                self.status(state["msg"])
                messagebox.showinfo("Exportar", f"{state['msg']}\n\n{path}")

        self.after(150, poll)

    def _tick(self):
        try:
            while True:
                try:
                    fn = self._queue.get_nowait()
                except queue.Empty:
                    break
                fn()
            now = time.perf_counter()
            if self.playing and not self.busy:
                self._advance()
            elif (not self.busy and self.last_frame_raw is not None and now - self._last_live > 0.04
                  and any(e.enabled and getattr(e, "LIVE_WHEN_PAUSED", False) for e in self.chain.effects)):
                self._last_live = now                      # p. ej. la cinta en pausa sigue «temblando»
                self._process_show(self.last_frame_raw, self.cur_idx)
            if now - self._last_meter > 0.1:
                self._last_meter = now
                self._update_side()
        except Exception:
            traceback.print_exc()
        self.after(4 if self.playing else 30, self._tick)
