"""Construcción de la interfaz (diseño minimalista): barra superior, vista previa, cadena de efectos y pestañas."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.params import FUENTES
from core.pipeline import by_category
from core.presets import PRESETS
from ui import theme as T
from ui.widgets import (EffectPicker, FlatButton, Meter, ParamSlider, Pills, ScrollFrame, Segmented, TimelineBar, Toggle,
                        section)

RESOLUCIONES = ["480", "640", "960", "1280", "1920", "Original"]
TABS = ["Efectos", "Audio", "Cinta", "Caras", "Salida"]
FACE_HEADERS = {
    "todas": "Modo «todas»: se censuran todas las caras (las casillas no cambian nada).",
    "todas menos las elegidas": "Activa las caras que quieres RESPETAR (sin censura):",
    "solo las elegidas": "Activa las caras que quieres CENSURAR:",
}
DESCRIPCIONES = {
    "Pantalla CRT": "Tubo de rayos catódicos: líneas, máscara de fósforo, curvatura, resplandor y presets.",
    "Cinta 8mm: ruido de pausa y rebobinado": "Ruido analógico que aparece solo al pausar, rebobinar o avanzar rápido.",
    "Cámara analógica (VHS / Hi8)": "Aspecto permanente de videocámara analógica con REC y fecha.",
    "Datamosh: Pixel-bending": "Arrastra los píxeles con el movimiento: 4 modos (arrastre, stretch, fluid, pack).",
    "Datamosh: Bloom (repetir delta)": "Repite el mismo movimiento una y otra vez; se dispara con beat, botón o movimiento.",
    "Caleidoscopio": "Segmentos espejo con torsión, ondulación y color por segmento.",
    "Mandala": "Capas de caleidoscopio con anillos y contra-giro.",
    "Warp de fluido (ruido)": "Deformación líquida con ruido fractal (domain warping).",
    "Plasma (generador)": "Plasma generativo con paletas; mézclalo con modos de composición.",
    "Slit-scan (Time Machine)": "Cada fila o columna viene de un instante distinto.",
    "Bloom (resplandor)": "Resplandor de las zonas brillantes, multi-escala y con tinte.",
    "Lente cinematográfica": "Aberración, distorsión, viñeta, halación y grano.",
    "Halftone (retícula)": "Puntos de imprenta en blanco y negro, color o CMY.",
    "Nube de puntos": "Partículas y curvas de nivel con aberración cromática.",
    "Estroboscopio / Flash": "Destellos con el beat, el BPM o una frecuencia fija.",
    "Vórtice (remolino)": "Remolino con pellizco y dispersión cromática.",
    "Censura de caras": "Pixelado o desenfoque automático; elige qué caras censurar.",
}


class PanelsMixin:
    # ===================================================================== construcción
    def _build_ui(self):
        self.configure(bg=T.BG)
        T.init_fonts(self)
        T.apply_style(self)
        self._build_topbar()
        self._build_status()
        self._build_body()

    def _build_topbar(self):
        bar = tk.Frame(self, bg=T.BG, height=50)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        right = tk.Frame(bar, bg=T.BG)
        right.pack(side="right", padx=(0, 12))
        self.btn_project = FlatButton(right, "Proyectar", self.toggle_projector, "solid")
        self.btn_project.pack(side="left", padx=3)
        mb = FlatButton(right, "Presets ▾", None, "ghost")
        mb.pack(side="left", padx=3)
        mb.command = lambda: self._popup_presets(mb)
        FlatButton(right, "Exportar", self.export, "primary").pack(side="left", padx=(6, 0))
        left = tk.Frame(bar, bg=T.BG)
        left.pack(side="left", padx=(16, 0))
        tk.Label(left, text="◐", fg=T.ACCENT, bg=T.BG, font=(T.FAMILY, 17)).pack(side="left")
        tk.Label(left, text="Editor", fg=T.TEXT, bg=T.BG, font=T.FONT_BOLD).pack(side="left", padx=(6, 18))
        FlatButton(left, "Abrir video", self.open_video, "ghost").pack(side="left")
        FlatButton(left, "Cámara", self.open_camera_dialog, "ghost").pack(side="left", padx=2)
        mid = tk.Frame(bar, bg=T.BG)
        mid.pack(side="left", expand=True)
        self.btn_rew = FlatButton(mid, "◀◀", lambda: self.toggle_search(-1), "icon")
        self.btn_rew.pack(side="left")
        self.btn_play = FlatButton(mid, "▶", self.toggle_play, "primary")
        self.btn_play.configure(width=4)
        self.btn_play.pack(side="left", padx=6)
        self.btn_ff = FlatButton(mid, "▶▶", lambda: self.toggle_search(1), "icon")
        self.btn_ff.pack(side="left")
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        tk.Label(mid, textvariable=self.time_var, fg=T.MUTED, bg=T.BG, font=T.FONT_MONO, width=14).pack(side="left", padx=(14, 0))

    def _popup_presets(self, widget):
        menu = tk.Menu(self, tearoff=0)
        groups = {"Cinta y CRT": [], "Personas y objetos": [], "Visuales": []}
        for name in PRESETS:
            key = "Cinta y CRT" if ("Cinta" in name or "CRT" in name) else (
                "Personas y objetos" if any(k in name for k in ("Personas", "Biblioteca", "Detección", "Privacidad", "Traqueo")) else "Visuales")
            groups[key].append(name)
        for g, names in groups.items():
            sub = tk.Menu(menu, tearoff=0)
            for n in names:
                sub.add_command(label=n, command=lambda n=n: self.apply_preset(n))
            menu.add_cascade(label=g, menu=sub)
        menu.tk_popup(widget.winfo_rootx(), widget.winfo_rooty() + widget.winfo_height())

    def _build_status(self):
        bar = tk.Frame(self, bg=T.BG, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Abre un video o elige una cámara para empezar.")
        tk.Label(bar, textvariable=self.status_var, fg=T.MUTED, bg=T.BG, font=T.FONT_SMALL, anchor="w").pack(side="left", padx=16)
        self.fps_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.fps_var, fg=T.FAINT, bg=T.BG, font=T.FONT_MONO).pack(side="right", padx=16)
        self.proj_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.proj_var, fg=T.GOOD, bg=T.BG, font=T.FONT_SMALL).pack(side="right", padx=8)

    def _build_body(self):
        body = tk.Frame(self, bg=T.BG)
        body.pack(fill="both", expand=True)
        stage = tk.Frame(body, bg=T.BG)
        stage.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=(0, 6))
        self.canvas = tk.Canvas(stage, bg="#050507", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Double-Button-1>", lambda e: self.toggle_projector())
        self.seek = TimelineBar(stage)
        self.seek.pack(fill="x", pady=(8, 0))
        self.seek.bind("<ButtonPress-1>", lambda e: setattr(self, "dragging", True), add="+")
        self.seek.bind("<ButtonRelease-1>", self._on_seek_release, add="+")
        side = tk.Frame(body, bg=T.PANEL, width=392)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)
        self.tab_bar = Segmented(side, TABS, command=self._show_tab, value="Efectos")
        self.tab_bar.configure(bg=T.PANEL)
        self.tab_bar.pack(fill="x", padx=8, pady=(8, 0))
        for b in self.tab_bar.btns.values():
            b.configure(bg=T.PANEL)
        tk.Frame(side, bg=T.BORDER, height=1).pack(fill="x")
        holder = tk.Frame(side, bg=T.PANEL)
        holder.pack(fill="both", expand=True)
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        self.tabs = {}
        for name in TABS:
            f = tk.Frame(holder, bg=T.PANEL)
            f.grid(row=0, column=0, sticky="nsew")
            self.tabs[name] = f
        self._build_effects_tab(self.tabs["Efectos"])
        self._build_audio_tab(self.tabs["Audio"])
        self._build_tape_tab(self.tabs["Cinta"])
        self._build_faces_tab(self.tabs["Caras"])
        self._build_output_tab(self.tabs["Salida"])
        self._show_tab("Efectos")

    def _show_tab(self, name):
        self.tabs[name].tkraise()
        self.tab_bar.select(name)
        self.current_tab = name

    # ===================================================================== pestaña Efectos
    def _build_effects_tab(self, tab):
        tab.configure(padx=12, pady=10)
        top = tk.Frame(tab, bg=T.PANEL)
        top.pack(fill="x")
        FlatButton(top, "+  Agregar efecto", self.open_picker, "primary").pack(side="left")
        FlatButton(top, "Vaciar", self._clear_chain, "ghost").pack(side="right")
        self.chain_scroll = ScrollFrame(tab, bg=T.PANEL)
        self.chain_scroll.pack(fill="x", pady=(10, 0))
        self.chain_scroll.cv.configure(height=170)
        self.sel_index = None
        self.err_var = tk.StringVar()
        tk.Label(tab, textvariable=self.err_var, fg=T.DANGER, bg=T.PANEL, font=T.FONT_SMALL, wraplength=350, justify="left",
                 anchor="w").pack(fill="x", pady=(6, 0))
        tk.Frame(tab, bg=T.BORDER, height=1).pack(fill="x", pady=(8, 8))
        self.pscroll = ScrollFrame(tab, bg=T.PANEL)
        self.pscroll.pack(fill="both", expand=True)
        self._rng = np.random.default_rng()

    def open_picker(self):
        EffectPicker(self, by_category(), self.add_effect, DESCRIPCIONES)

    def _clear_chain(self):
        self.chain.clear()
        self.sel_index = None
        self._refresh_list()
        self._request_refresh()

    def _cur_index(self):
        i = self.sel_index
        return i if i is not None and 0 <= i < len(self.chain.effects) else None

    def remove_effect(self):
        i = self._cur_index()
        if i is None:
            return
        self.chain.remove(i)
        self._refresh_list(select=min(i, len(self.chain.effects) - 1) if self.chain.effects else None)
        self._request_refresh()

    def move_effect(self, delta, index=None):
        i = self._cur_index() if index is None else index
        if i is None:
            return
        self._refresh_list(select=self.chain.move(i, delta))
        self._request_refresh()

    def _select_effect(self, i):
        self._refresh_list(select=i)

    def _set_enabled(self, i, value):
        if 0 <= i < len(self.chain.effects):
            self.chain.effects[i].enabled = bool(value)
            self._request_refresh()

    def _refresh_list(self, select=None):
        if select is not None:
            self.sel_index = select
        if self.sel_index is not None and self.sel_index >= len(self.chain.effects):
            self.sel_index = len(self.chain.effects) - 1 if self.chain.effects else None
        sf = self.chain_scroll
        sf.clear()
        if not self.chain.effects:
            tk.Label(sf.inner, text="Sin efectos todavía.\nPulsa «Agregar efecto» o elige un preset.", fg=T.FAINT, bg=T.PANEL,
                     font=T.FONT_SMALL, justify="left").pack(anchor="w", pady=14, padx=4)
        for i, e in enumerate(self.chain.effects):
            sel = i == self.sel_index
            bg = T.ACCENT_DIM if sel else T.CARD
            row = tk.Frame(sf.inner, bg=bg, padx=8, pady=6, cursor="hand2")
            row.pack(fill="x", pady=2, padx=(0, 6))
            tg = Toggle(row, e.enabled, command=lambda v, i=i: self._set_enabled(i, v))
            tg.configure(bg=bg)
            tg.pack(side="left")
            name = tk.Label(row, text=e.name + ("  ·  siempre primero" if e.pin_first else ""), fg=T.TEXT if e.enabled else T.FAINT,
                            bg=bg, font=T.FONT, anchor="w", cursor="hand2")
            name.pack(side="left", padx=(10, 0), fill="x", expand=True)
            for txt, cmd in (("✕", lambda i=i: (self._select_effect(i), self.remove_effect())),
                             ("▼", lambda i=i: self.move_effect(1, i)), ("▲", lambda i=i: self.move_effect(-1, i))):
                b = tk.Label(row, text=txt, fg=T.MUTED, bg=bg, font=T.FONT_SMALL, padx=4, cursor="hand2")
                b.pack(side="right")
                b.bind("<Button-1>", lambda ev, c=cmd: c())
                b.bind("<Enter>", lambda ev, w=b: w.configure(fg=T.TEXT))
                b.bind("<Leave>", lambda ev, w=b: w.configure(fg=T.MUTED))
            for w in (row, name):
                w.bind("<Button-1>", lambda ev, i=i: self._select_effect(i))
        cur = self._cur_index()
        self._show_params(self.chain.effects[cur] if cur is not None else None)

    # ---------------------------------------------------------------- parámetros del efecto
    def _show_params(self, effect):
        self.selected_effect = effect
        ps = self.pscroll
        ps.clear()
        inner = ps.inner
        if effect is None:
            tk.Label(inner, text="Selecciona un efecto para ver sus controles.", fg=T.FAINT, bg=T.PANEL, font=T.FONT_SMALL).pack(
                anchor="w", pady=10)
            return
        head = tk.Frame(inner, bg=T.PANEL)
        head.pack(fill="x", pady=(0, 4), padx=(0, 6))
        tk.Label(head, text=effect.name, fg=T.TEXT, bg=T.PANEL, font=T.FONT_BOLD, wraplength=230, justify="left", anchor="w").pack(side="left")
        FlatButton(head, "↺", lambda: self._reset_effect(effect), "icon").pack(side="right")
        FlatButton(head, "Aleatorio", lambda: self._randomize_effect(effect), "solid").pack(side="right", padx=4)
        body = tk.Frame(inner, bg=T.PANEL)
        body.pack(fill="x", padx=(0, 10))
        for p in effect.params.values():
            if not p.key.startswith("_"):
                self._param_widget(body, effect, p)
        if "_mix" in effect.params:
            section(body, "Mezcla").pack(fill="x", pady=(8, 8))
            self._param_widget(body, effect, effect.params["_mix"])
            self._param_widget(body, effect, effect.params["_blend"], combo=True)

    def _reset_effect(self, effect):
        effect.reset_params()
        self._show_params(effect)
        self._request_refresh()

    def _randomize_effect(self, effect):
        effect.randomize(self._rng, 0.5)
        self._show_params(effect)
        self._request_refresh()

    def _param_widget(self, parent, effect, p, combo=False):
        k = p.kind
        if k == "button":
            FlatButton(parent, p.label, lambda: self._press_button(effect, p.key), "solid").pack(fill="x", pady=(2, 10))
        elif k == "bool":
            row = tk.Frame(parent, bg=T.PANEL)
            row.pack(fill="x", pady=(0, 10))
            tk.Label(row, text=p.label, fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL, wraplength=280, justify="left", anchor="w").pack(side="left")
            Toggle(row, p.value, command=lambda v: (effect.set_value(p.key, v), self._request_refresh())).pack(side="right")
        elif k == "choice":
            self._param_choice(parent, effect, p, combo)
        elif k == "text":
            tk.Label(parent, text=p.label, fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL, wraplength=330, justify="left", anchor="w").pack(fill="x")
            var = tk.StringVar(value=str(p.value))
            ent = ttk.Entry(parent, textvariable=var)
            ent.pack(fill="x", pady=(2, 10))
            apply = lambda e=None: (effect.set_value(p.key, var.get()), self._request_refresh())  # noqa: E731
            ent.bind("<Return>", apply)
            ent.bind("<FocusOut>", apply)
            ent._var = var
        else:
            self._param_numeric(parent, effect, p)

    def _press_button(self, effect, key):
        effect.on_button(key)
        self._show_params(effect)                # por si el botón cambió valores (p. ej. un preset)
        self._request_refresh()

    def _param_choice(self, parent, effect, p, combo=False):
        tk.Label(parent, text=p.label, fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL, anchor="w").pack(fill="x")
        helps = (getattr(effect, "HELP", {}) or {}).get(p.key)
        hvar = tk.StringVar(value=(helps or {}).get(str(p.value), ""))

        def pick(v):
            effect.set_value(p.key, v)
            hvar.set((helps or {}).get(v, ""))
            self._request_refresh()

        if len(p.choices) <= 3 and sum(len(c) for c in p.choices) < 34 and not combo:
            w = Pills(parent, p.choices, p.value, pick)
            w.pack(anchor="w", pady=(3, 2))
        else:
            var = tk.StringVar(value=str(p.value))
            cb = ttk.Combobox(parent, textvariable=var, values=p.choices, state="readonly")
            cb.pack(fill="x", pady=(3, 2))
            cb.bind("<<ComboboxSelected>>", lambda e: pick(var.get()))
            cb._var = var
        if helps:
            tk.Label(parent, textvariable=hvar, fg=T.FAINT, bg=T.PANEL, font=T.FONT_SMALL, wraplength=330, justify="left", anchor="w").pack(fill="x")
        tk.Frame(parent, bg=T.PANEL, height=10).pack()
        parent._keep = getattr(parent, "_keep", []) + [hvar]

    def _param_numeric(self, parent, effect, p):
        holder = tk.Frame(parent, bg=T.PANEL)
        holder.pack(fill="x", pady=(0, 8))
        top = tk.Frame(holder, bg=T.PANEL)
        top.pack(fill="x")
        span = p.hi - p.lo

        def fmt(v):
            if p.kind == "int":
                return str(int(round(v)))
            return f"{v:.3f}" if span <= 0.2 else (f"{v:.2f}" if span <= 10 else f"{v:.1f}")

        def on_change(v):
            effect.set_value(p.key, v)
            self._request_refresh()

        sl = ParamSlider(top, p.label, p.lo, p.hi, p.value, p.default, on_change, fmt=fmt, integer=(p.kind == "int"),
                         modulated=p.mod_source != "ninguna")
        sl.pack(side="left", fill="x", expand=True)
        if p.key.startswith("_"):                         # los controles de mezcla no se modulan desde aquí
            return
        modrow = tk.Frame(holder, bg=T.PANEL)
        src = tk.StringVar(value=p.mod_source)
        cb = ttk.Combobox(modrow, textvariable=src, values=FUENTES, state="readonly", width=10)
        cb.pack(side="left", padx=(0, 8), pady=(2, 0))
        amt = ParamSlider(modrow, "cantidad", -1, 1, p.mod_amount, 0.0, lambda v: apply_mod(), fmt=lambda v: f"{v:+.2f}")
        amt.pack(side="left", fill="x", expand=True)

        def apply_mod(*_):
            effect.set_mod(p.key, src.get(), float(amt.value))
            sl.set_modulated(src.get() != "ninguna" and abs(amt.value) > 0.001)
            self._request_refresh()

        cb.bind("<<ComboboxSelected>>", apply_mod)
        shown = {"v": p.mod_source != "ninguna"}
        if shown["v"]:
            modrow.pack(fill="x")
        btn = FlatButton(top, "⚡", None, "icon")
        btn.pack(side="right", padx=(6, 0))

        def toggle():
            shown["v"] = not shown["v"]
            btn.set_active(shown["v"])
            modrow.pack(fill="x") if shown["v"] else modrow.pack_forget()

        btn.command = toggle
        btn.set_active(shown["v"])
        holder._keep = [src]

    # ===================================================================== pestaña Audio
    def _build_audio_tab(self, tab):
        tab.configure(padx=14, pady=12)
        section(tab, "Fuente de las señales").pack(fill="x", pady=(0, 8))
        self.audio_src = tk.StringVar(value="Del video")
        self._audio_pills = Pills(tab, ["Del video", "Micrófono", "Ninguno"], "Del video",
                                  lambda v: (self.audio_src.set(v), self._on_audio_src()))
        self._audio_pills.pack(anchor="w")
        self.audio_note = tk.StringVar(value="")
        tk.Label(tab, textvariable=self.audio_note, fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL, wraplength=350, justify="left",
                 anchor="w").pack(fill="x", pady=(6, 0))
        row = tk.Frame(tab, bg=T.PANEL)
        row.pack(fill="x", pady=(10, 0))
        tk.Label(row, text="Silenciar la vista previa", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(side="left")
        self.mute_var = tk.BooleanVar(value=False)
        Toggle(row, False, command=lambda v: (self.mute_var.set(v), self._on_mute())).pack(side="right")
        section(tab, "Ritmo").pack(fill="x", pady=(16, 8))
        grid = tk.Frame(tab, bg=T.PANEL)
        grid.pack(fill="x")
        self.bpm_var, self.cycle_var = tk.StringVar(value="120"), tk.StringVar(value="4")
        for r, (lbl, var, lo, hi) in enumerate((("BPM", self.bpm_var, 40, 240), ("Beats por ciclo del oscilador", self.cycle_var, 1, 16))):
            tk.Label(grid, text=lbl, fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).grid(row=r, column=0, sticky="w", pady=3)
            sp = ttk.Spinbox(grid, from_=lo, to=hi, textvariable=var, width=6, command=self._on_bpm)
            sp.grid(row=r, column=1, sticky="e", padx=(12, 0))
            sp.bind("<Return>", lambda e: self._on_bpm())
            sp.bind("<FocusOut>", lambda e: self._on_bpm())
        grid.grid_columnconfigure(0, weight=1)
        self.sens_var = tk.DoubleVar(value=1.45)
        ParamSlider(tab, "Sensibilidad del beat (menor = más sensible)", 1.1, 2.5, 1.45, 1.45,
                    lambda v: (self.sens_var.set(v), self._on_sens(v))).pack(fill="x", pady=(10, 0))
        section(tab, "Señales").pack(fill="x", pady=(16, 8))
        self.meters = {}
        for name, col in (("graves", T.ACCENT), ("medios", T.GOOD), ("agudos", T.WARN), ("volumen", T.TEXT), ("beat", T.DANGER)):
            r = tk.Frame(tab, bg=T.PANEL)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=name.capitalize(), fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL, width=9, anchor="w").pack(side="left")
            m = Meter(r, color=col)
            m.pack(side="left", fill="x", expand=True)
            self.meters[name] = m
        self.spec_canvas = tk.Canvas(tab, height=80, bg=T.CARD, highlightthickness=0)
        self.spec_canvas.pack(fill="x", pady=(12, 0))
        tk.Label(tab, text="Cada parámetro tiene un botón ⚡ para enlazarlo a una señal o a un oscilador sincronizado con el BPM.",
                 fg=T.FAINT, bg=T.PANEL, font=T.FONT_SMALL, wraplength=350, justify="left").pack(anchor="w", pady=(12, 0))

    # ===================================================================== pestaña Cinta
    def _build_tape_tab(self, tab):
        tab.configure(padx=14, pady=12)
        tk.Label(tab, text="Con el efecto «Cinta 8mm» en la cadena, el ruido aparece al pausar, rebobinar (◀◀) o avanzar rápido (▶▶). "
                           "Para dejarlo en el video exportado, inserta eventos en el guion.", fg=T.MUTED, bg=T.PANEL,
                 font=T.FONT_SMALL, wraplength=350, justify="left").pack(anchor="w")
        r = tk.Frame(tab, bg=T.PANEL)
        r.pack(fill="x", pady=(10, 0))
        tk.Label(r, text="Velocidad de ◀◀ / ▶▶", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(side="left")
        self.search_speed = tk.StringVar(value="8")
        ttk.Combobox(r, textvariable=self.search_speed, values=["2", "4", "8", "16", "32"], width=5, state="readonly").pack(side="right")
        section(tab, "Guion de cinta").pack(fill="x", pady=(16, 8))
        b = tk.Frame(tab, bg=T.PANEL)
        b.pack(fill="x")
        FlatButton(b, "+ Pausa", lambda: self.add_event("pausa"), "solid").pack(side="left")
        FlatButton(b, "+ Rebobinar", lambda: self.add_event("rebobinar"), "solid").pack(side="left", padx=4)
        FlatButton(b, "+ Avance", lambda: self.add_event("avance"), "solid").pack(side="left")
        self.ev_list = tk.Listbox(tab, height=8, bg=T.CARD, fg=T.TEXT, selectbackground=T.ACCENT_DIM, selectforeground=T.TEXT,
                                  activestyle="none", exportselection=False, highlightthickness=0, borderwidth=0, font=T.FONT_SMALL)
        self.ev_list.pack(fill="x", pady=8)
        r2 = tk.Frame(tab, bg=T.PANEL)
        r2.pack(fill="x")
        for txt, cmd in (("Quitar", self.remove_event), ("Quitar todo", self.clear_events), ("Guardar", self.save_events), ("Cargar", self.load_events)):
            FlatButton(r2, txt, cmd, "ghost").pack(side="left")
        r3 = tk.Frame(tab, bg=T.PANEL)
        r3.pack(fill="x", pady=(12, 0))
        tk.Label(r3, text="Pausa al final de cada rebobinado (s)", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(side="left")
        self.hold_var = tk.StringVar(value="0.6")
        ttk.Spinbox(r3, from_=0.1, to=5.0, increment=0.1, textvariable=self.hold_var, width=5).pack(side="right")
        FlatButton(tab, "▶  Previsualizar el guion (sin sonido)", self.preview_plan, "solid").pack(fill="x", pady=(14, 0))
        tk.Label(tab, text="Lo rebobinado se vuelve a reproducir, como en una cinta real. El sonido solo suena en la reproducción normal.",
                 fg=T.FAINT, bg=T.PANEL, font=T.FONT_SMALL, wraplength=350, justify="left").pack(anchor="w", pady=(10, 0))

    def _refresh_events(self):
        self.events.sort(key=lambda e: e["frame"])
        self.ev_list.delete(0, "end")
        for ev in self.events:
            self.ev_list.insert("end", "  " + self._event_text(ev))
        col = {"pausa": T.WARN, "rebobinar": T.DANGER, "avance": T.GOOD}
        self.seek.set_markers([(ev["frame"], col.get(ev["tipo"], T.WARN)) for ev in self.events])

    # ===================================================================== pestaña Caras
    def _build_faces_tab(self, tab):
        tab.configure(padx=14, pady=12)
        tk.Label(tab, text="Agrega «Censura de caras» (Agregar efecto → Caras). Elige qué caras censurar con los interruptores "
                           "o haciendo clic sobre la cara en la vista previa.", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL,
                 wraplength=350, justify="left").pack(anchor="w")
        r = tk.Frame(tab, bg=T.PANEL)
        r.pack(fill="x", pady=(10, 0))
        tk.Label(r, text="Mostrar recuadros en la vista previa", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(side="left")
        self.overlay_var = tk.BooleanVar(value=True)
        Toggle(r, True, command=lambda v: (self.overlay_var.set(v), self._redraw())).pack(side="right")
        self.faces_hdr = tk.StringVar(value=FACE_HEADERS["todas"])
        tk.Label(tab, textvariable=self.faces_hdr, fg=T.TEXT, bg=T.PANEL, font=T.FONT_BOLD, wraplength=350, justify="left",
                 anchor="w").pack(fill="x", pady=(14, 6))
        self.faces_scroll = ScrollFrame(tab, bg=T.PANEL)
        self.faces_scroll.pack(fill="both", expand=True)
        self.faces_frame = self.faces_scroll.inner
        b = tk.Frame(tab, bg=T.PANEL)
        b.pack(fill="x", pady=(8, 0))
        FlatButton(b, "Marcar todas", self._faces_all, "solid").pack(side="left")
        FlatButton(b, "Desmarcar", self._faces_none, "ghost").pack(side="left", padx=4)
        b2 = tk.Frame(tab, bg=T.PANEL)
        b2.pack(fill="x", pady=(6, 0))
        FlatButton(b2, "Foto de referencia…", self.load_reference, "solid").pack(side="left")
        FlatButton(b2, "Borrar referencias", self._clear_refs, "ghost").pack(side="left", padx=4)
        tk.Label(tab, text="Con una foto de referencia se reconoce a la persona aunque salga y vuelva a entrar.", fg=T.FAINT, bg=T.PANEL,
                 font=T.FONT_SMALL, wraplength=350, justify="left").pack(anchor="w", pady=(8, 0))

    def _refresh_faces(self):
        self.faces_hdr.set(FACE_HEADERS.get(self.faces.mode, FACE_HEADERS["todas"]))
        known = self.faces.known_list()
        sig = tuple(k for k, _, _ in known)
        if sig != self._faces_sig:
            self.faces_scroll.clear()
            self._face_vars, self._face_imgs, self._face_toggles = {}, [], {}
            if not known:
                tk.Label(self.faces_frame, text="Todavía no se ha detectado ninguna cara.", fg=T.FAINT, bg=T.PANEL,
                         font=T.FONT_SMALL).pack(anchor="w", pady=6)
            for key, thumb, sel in known:
                r = tk.Frame(self.faces_frame, bg=T.CARD, padx=8, pady=6)
                r.pack(fill="x", pady=3, padx=(0, 6))
                im = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)))
                self._face_imgs.append(im)
                tk.Label(r, image=im, bg=T.CARD).pack(side="left")
                tk.Label(r, text=key, fg=T.TEXT, bg=T.CARD, font=T.FONT).pack(side="left", padx=10)
                var = tk.BooleanVar(value=sel)
                self._face_vars[key] = var
                tg = Toggle(r, sel, command=lambda v, k=key, var=var: (var.set(v), self._set_face_selected(k, v)))
                tg.configure(bg=T.CARD)
                tg.pack(side="right")
                self._face_toggles[key] = tg
            self._faces_sig = sig
        for key, _, sel in known:
            var = self._face_vars.get(key)
            if var is not None and var.get() != sel:
                var.set(sel)
                self._face_toggles[key].set(sel)

    # ===================================================================== pestaña Salida
    def _build_output_tab(self, tab):
        tab.configure(padx=14, pady=12)
        sf = ScrollFrame(tab, bg=T.PANEL)
        sf.pack(fill="both", expand=True)
        p = sf.inner
        section(p, "Proyector").pack(fill="x", pady=(0, 8), padx=(0, 8))
        tk.Label(p, text="Abre una ventana aparte con solo la imagen, para llevarla a un proyector o pantalla grande.", fg=T.MUTED,
                 bg=T.PANEL, font=T.FONT_SMALL, wraplength=340, justify="left").pack(anchor="w")
        self.btn_proj2 = FlatButton(p, "Abrir ventana de proyección", self.toggle_projector, "primary")
        self.btn_proj2.pack(fill="x", pady=(10, 4), padx=(0, 8))
        FlatButton(p, "Abrir en pantalla completa", self.projector_fullscreen, "solid").pack(fill="x", pady=(0, 4), padx=(0, 8))
        self.monitor_var = tk.StringVar(value="Pantalla actual")
        tk.Label(p, text="Monitor de destino", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(anchor="w", pady=(6, 2))
        self.monitor_cb = ttk.Combobox(p, textvariable=self.monitor_var, values=["Pantalla actual"], state="readonly")
        self.monitor_cb.pack(fill="x", padx=(0, 8))
        self.monitor_cb.bind("<<ComboboxSelected>>", lambda e: self._on_monitor())
        tk.Label(p, text="Ajuste de la imagen", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(anchor="w", pady=(10, 2))
        self.fit_pills = Pills(p, ["ajustar", "llenar", "estirar"], "ajustar", self._on_fit)
        self.fit_pills.pack(anchor="w")
        tk.Label(p, text="En la ventana: F o F11 = pantalla completa · Esc = salir · B = negro · M = ajuste", fg=T.FAINT, bg=T.PANEL,
                 font=T.FONT_SMALL, wraplength=340, justify="left").pack(anchor="w", pady=(8, 0))
        self.save_cpu_var = tk.BooleanVar(value=False)
        self._toggle_row(p, "Ahorrar CPU: no dibujar la vista previa principal mientras se proyecta", False, lambda v: self.save_cpu_var.set(v))

        section(p, "Cámara").pack(fill="x", pady=(18, 8), padx=(0, 8))
        self.cam_label = tk.StringVar(value="Sin cámara activa")
        tk.Label(p, textvariable=self.cam_label, fg=T.TEXT, bg=T.PANEL, font=T.FONT_SMALL).pack(anchor="w")
        FlatButton(p, "Elegir o cambiar cámara…", self.open_camera_dialog, "solid").pack(anchor="w", pady=6)

        section(p, "Rendimiento").pack(fill="x", pady=(14, 8), padx=(0, 8))
        r = tk.Frame(p, bg=T.PANEL)
        r.pack(fill="x", padx=(0, 8))
        tk.Label(r, text="Resolución de trabajo (px de ancho)", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(side="left")
        self.res_var = tk.StringVar(value="640")
        cb = ttk.Combobox(r, textvariable=self.res_var, values=RESOLUCIONES, width=9, state="readonly")
        cb.pack(side="right")
        cb.bind("<<ComboboxSelected>>", lambda e: self._reprocess_current())
        tk.Label(p, text="Se usa en la vista previa, la proyección y la exportación. Para proyectar en grande usa 1280 o más si tu equipo aguanta.",
                 fg=T.FAINT, bg=T.PANEL, font=T.FONT_SMALL, wraplength=340, justify="left").pack(anchor="w", pady=(4, 0))
        self.loop_var = tk.BooleanVar(value=True)
        self._toggle_row(p, "Repetir el video al terminar", True, lambda v: self.loop_var.set(v))

        section(p, "Auto-VJ").pack(fill="x", pady=(18, 8), padx=(0, 8))
        tk.Label(p, text="Cambia de preset automáticamente (útil en directo). No toca la censura de caras.", fg=T.MUTED, bg=T.PANEL,
                 font=T.FONT_SMALL, wraplength=340, justify="left").pack(anchor="w")
        self.vj_on = tk.BooleanVar(value=False)
        self._toggle_row(p, "Activar Auto-VJ", False, lambda v: self._on_vj(v))
        r = tk.Frame(p, bg=T.PANEL)
        r.pack(fill="x", pady=4, padx=(0, 8))
        tk.Label(r, text="Cambiar cada (segundos)", fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL).pack(side="left")
        self.vj_secs = tk.StringVar(value="12")
        ttk.Spinbox(r, from_=2, to=120, textvariable=self.vj_secs, width=6).pack(side="right")
        self.vj_beat = tk.BooleanVar(value=False)
        self._toggle_row(p, "Cambiar con el beat (cada 16 golpes)", False, lambda v: self.vj_beat.set(v))
        FlatButton(p, "Siguiente preset ahora", self.vj_next, "solid").pack(anchor="w", pady=6)

        section(p, "Herramientas").pack(fill="x", pady=(18, 8), padx=(0, 8))
        FlatButton(p, "Datamosh real (archivo)…", self.open_datamosh_tool, "solid").pack(anchor="w")
        FlatButton(p, "Biblioteca de imágenes…", self.open_library, "ghost").pack(anchor="w", pady=2)
        FlatButton(p, "Guardar un fotograma", self.save_frame, "ghost").pack(anchor="w")
        tk.Label(p, text="Atajos: Espacio reproducir · J rebobinar · L avanzar · K pausar · P proyectar · B negro · R aleatorio · Ctrl+E exportar",
                 fg=T.FAINT, bg=T.PANEL, font=T.FONT_SMALL, wraplength=340, justify="left").pack(anchor="w", pady=(16, 8))

    def _toggle_row(self, parent, text, value, command):
        r = tk.Frame(parent, bg=T.PANEL)
        r.pack(fill="x", pady=5, padx=(0, 8))
        tk.Label(r, text=text, fg=T.MUTED, bg=T.PANEL, font=T.FONT_SMALL, wraplength=270, justify="left", anchor="w").pack(side="left")
        Toggle(r, value, command=command).pack(side="right")

    # ===================================================================== actualización periódica
    def _update_side(self):
        a = self.ctx.audio
        if self.current_tab == "Audio":
            for k, pb in self.meters.items():
                pb["value"] = float(a.get(k, 0.0))
            self.spec_canvas.delete("all")
            spec = a.get("spectrum")
            if spec is not None:
                w = max(self.spec_canvas.winfo_width(), 10)
                bw = w / len(spec)
                for i, v in enumerate(spec):
                    hgt = 3 + float(v) * 70
                    self.spec_canvas.create_rectangle(i * bw + 1, 76 - hgt, (i + 1) * bw - 1, 76, fill=T.ACCENT, width=0)
        e = self.selected_effect
        self.err_var.set(e.error if e is not None else "")
        self._meter_count += 1
        if self._meter_count % 3 == 0 and self.current_tab == "Caras":
            self._refresh_faces()
