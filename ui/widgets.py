"""Componentes visuales propios (planos y minimalistas) para la interfaz."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui import theme as T


class FlatButton(tk.Label):
    """Botón plano. estilos: ghost (transparente), solid (relleno), primary (acento), icon."""

    def __init__(self, master, text="", command=None, style="ghost", width=None, **kw):
        self.style, self.command, self.active, self._disabled = style, command, False, False
        bg = master.cget("bg") if "bg" in master.keys() else T.PANEL
        self.base_bg = {"ghost": bg, "solid": T.CARD, "primary": T.ACCENT, "icon": bg}[style]
        self.hover_bg = {"ghost": T.CARD_HOVER, "solid": T.CARD_HOVER, "primary": T.ACCENT_HOVER, "icon": T.CARD_HOVER}[style]
        fg = "#ffffff" if style == "primary" else T.TEXT
        padx = 6 if style == "icon" else 12
        super().__init__(master, text=text, bg=self.base_bg, fg=fg, font=T.FONT, padx=padx, pady=5, cursor="hand2",
                         width=width or 0, **kw)
        self.bind("<Enter>", lambda e: self._paint(True))
        self.bind("<Leave>", lambda e: self._paint(False))
        self.bind("<ButtonRelease-1>", self._release)

    def _paint(self, hover):
        if self._disabled:
            return
        bg = self.hover_bg if hover else (T.ACCENT_DIM if self.active and self.style != "primary" else self.base_bg)
        fg = T.ACCENT_HOVER if self.active and self.style != "primary" else ("#ffffff" if self.style == "primary" else T.TEXT)
        super().configure(bg=bg, fg=fg)

    def _release(self, e):
        if self._disabled or not self.command:
            return
        if 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height():
            self.command()

    def set_active(self, on):
        self.active = on
        self._paint(False)

    def set_enabled(self, on):
        self._disabled = not on
        super().configure(fg=T.TEXT if on else T.FAINT, cursor="hand2" if on else "arrow")


class Toggle(tk.Canvas):
    """Interruptor tipo «switch»."""

    def __init__(self, master, value=False, command=None, **kw):
        super().__init__(master, width=34, height=18, bg=master.cget("bg"), highlightthickness=0, cursor="hand2", **kw)
        self.value, self.command = bool(value), command
        self.bind("<Button-1>", lambda e: self.toggle())
        self._draw()

    def _draw(self):
        self.delete("all")
        on = self.value
        self.create_oval(0, 1, 17, 17, fill=T.ACCENT if on else T.BORDER, outline="")
        self.create_oval(17, 1, 34, 17, fill=T.ACCENT if on else T.BORDER, outline="")
        self.create_rectangle(8, 1, 26, 17, fill=T.ACCENT if on else T.BORDER, outline="")
        x = 20 if on else 3
        self.create_oval(x, 4, x + 11, 15, fill="#ffffff" if on else T.MUTED, outline="")

    def set(self, v):
        self.value = bool(v)
        self._draw()

    def toggle(self):
        self.value = not self.value
        self._draw()
        if self.command:
            self.command(self.value)


class Segmented(tk.Frame):
    """Fila de opciones excluyentes (pestañas o interruptores de modo)."""

    def __init__(self, master, options, command=None, value=None, accent_bar=True, **kw):
        super().__init__(master, bg=master.cget("bg"), **kw)
        self.command, self.accent_bar = command, accent_bar
        self.btns = {}
        self.value = value if value is not None else options[0]
        for o in options:
            b = tk.Label(self, text=o, font=T.FONT, padx=12, pady=7, cursor="hand2", bg=self.cget("bg"))
            b.pack(side="left")
            b.bind("<Button-1>", lambda e, v=o: self.select(v, True))
            b.bind("<Enter>", lambda e, bb=b: bb.configure(fg=T.TEXT))
            b.bind("<Leave>", lambda e: self._paint())
            self.btns[o] = b
        self._paint()

    def _paint(self):
        for o, b in self.btns.items():
            on = o == self.value
            b.configure(fg=T.TEXT if on else T.MUTED, font=T.FONT_BOLD if on else T.FONT)

    def select(self, v, fire=False):
        self.value = v
        self._paint()
        if fire and self.command:
            self.command(v)


class Pills(tk.Frame):
    """Selector compacto de pocas opciones (2-4) dentro de una cápsula."""

    def __init__(self, master, options, value, command):
        super().__init__(master, bg=T.CARD, padx=2, pady=2)
        self.command, self.value, self.btns = command, value, {}
        for o in options:
            b = tk.Label(self, text=o, font=T.FONT_SMALL, padx=10, pady=4, cursor="hand2")
            b.pack(side="left", padx=1)
            b.bind("<Button-1>", lambda e, v=o: self.select(v, True))
            self.btns[o] = b
        self._paint()

    def _paint(self):
        for o, b in self.btns.items():
            on = o == self.value
            b.configure(bg=T.ACCENT if on else T.CARD, fg="#ffffff" if on else T.MUTED)

    def select(self, v, fire=False):
        self.value = v
        self._paint()
        if fire:
            self.command(v)


class ParamSlider(tk.Frame):
    """Deslizador minimalista: etiqueta a la izquierda, valor a la derecha, barra fina debajo.
    Doble clic = valor por defecto. Rueda del ratón = ajuste fino."""

    def __init__(self, master, label, lo, hi, value, default, command, fmt=None, integer=False, modulated=False):
        super().__init__(master, bg=master.cget("bg"))
        self.lo, self.hi, self.default, self.command = float(lo), float(hi), default, command
        self.integer, self.fmt = integer, fmt or (lambda v: f"{int(round(v))}" if integer else f"{v:.2f}")
        self.value = float(value)
        self.modulated = modulated
        top = tk.Frame(self, bg=self.cget("bg"))
        top.pack(fill="x")
        self.lbl = tk.Label(top, text=label, font=T.FONT_SMALL, fg=T.MUTED, bg=self.cget("bg"), anchor="w")
        self.lbl.pack(side="left")
        self.val = tk.Label(top, text="", font=T.FONT_SMALL, fg=T.TEXT, bg=self.cget("bg"))
        self.val.pack(side="right")
        self.cv = tk.Canvas(self, height=16, bg=self.cget("bg"), highlightthickness=0, cursor="sb_h_double_arrow")
        self.cv.pack(fill="x")
        self.cv.bind("<Configure>", lambda e: self._draw())
        self.cv.bind("<Button-1>", self._click)
        self.cv.bind("<B1-Motion>", self._click)
        self.cv.bind("<Double-Button-1>", lambda e: self.reset())
        for w in (self.cv, self.lbl, self.val):
            w.bind("<MouseWheel>", self._wheel)
            w.bind("<Button-4>", self._wheel)
            w.bind("<Button-5>", self._wheel)

    def _frac(self):
        return 0.0 if self.hi == self.lo else (self.value - self.lo) / (self.hi - self.lo)

    def _draw(self):
        c = self.cv
        c.delete("all")
        w = max(c.winfo_width(), 20)
        y = 8
        c.create_line(6, y, w - 6, y, fill=T.BORDER, width=3, capstyle="round")
        x = 6 + self._frac() * (w - 12)
        c.create_line(6, y, x, y, fill=T.GOOD if self.modulated else T.ACCENT, width=3, capstyle="round")
        c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#ffffff", outline="")
        self.val.configure(text=self.fmt(self.value))

    def _set(self, v, fire=True):
        v = min(max(v, self.lo), self.hi)
        if self.integer:
            v = float(round(v))
        self.value = v
        self._draw()
        if fire and self.command:
            self.command(int(v) if self.integer else v)

    def _click(self, e):
        w = max(self.cv.winfo_width() - 12, 1)
        self._set(self.lo + (min(max(e.x - 6, 0), w) / w) * (self.hi - self.lo))

    def _wheel(self, e):
        step = (self.hi - self.lo) / (20 if not self.integer else max(1, self.hi - self.lo))
        d = 1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else -1
        self._set(self.value + d * (1 if self.integer else step))
        return "break"

    def reset(self):
        self._set(float(self.default))

    def set(self, v):
        self.value = float(v)
        self._draw()

    def set_modulated(self, on):
        self.modulated = on
        self._draw()


class TimelineBar(tk.Canvas):
    """Línea de tiempo con cabezal y marcas de eventos. Compatible con la API mínima de ttk.Scale."""

    def __init__(self, master, **kw):
        super().__init__(master, height=22, bg=master.cget("bg"), highlightthickness=0, cursor="hand2", **kw)
        self.lo, self.hi, self.value, self.markers, self.disabled = 0.0, 1.0, 0.0, [], False
        self.dragging = False
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", self._down)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._up)

    def configure(self, cnf=None, **kw):
        if "from_" in kw:
            self.lo = float(kw.pop("from_"))
        if "to" in kw:
            self.hi = max(float(kw.pop("to")), self.lo + 1e-6)
        if kw or cnf:
            super().configure(cnf, **kw) if cnf else super().configure(**kw)
        self._draw()

    def state(self, flags):
        self.disabled = "disabled" in flags and "!disabled" not in flags
        self._draw()

    def set(self, v):
        if not self.dragging:
            self.value = float(v)
            self._draw()

    def get(self):
        return self.value

    def set_markers(self, markers):
        self.markers = markers
        self._draw()

    def _x(self, v):
        w = max(self.winfo_width(), 20)
        return 8 + (v - self.lo) / (self.hi - self.lo) * (w - 16)

    def _draw(self):
        self.delete("all")
        w, y = max(self.winfo_width(), 20), 11
        self.create_line(8, y, w - 8, y, fill=T.BORDER, width=3, capstyle="round")
        if self.disabled:
            return
        x = self._x(self.value)
        self.create_line(8, y, x, y, fill=T.ACCENT, width=3, capstyle="round")
        for v, col in self.markers:
            mx = self._x(v)
            self.create_rectangle(mx - 1, y - 7, mx + 1, y + 7, fill=col, outline="")
        self.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ffffff", outline="")

    def _val_at(self, x):
        w = max(self.winfo_width() - 16, 1)
        return self.lo + min(max((x - 8) / w, 0.0), 1.0) * (self.hi - self.lo)

    def _down(self, e):
        if self.disabled:
            return
        self.dragging = True
        self.value = self._val_at(e.x)
        self._draw()

    def _move(self, e):
        if self.dragging:
            self.value = self._val_at(e.x)
            self._draw()

    def _up(self, e):
        if self.dragging:
            self.value = self._val_at(e.x)
            self.dragging = False
            self._draw()


class Meter(tk.Canvas):
    """Barra de nivel. Se asigna con meter['value'] = x (0..1)."""

    def __init__(self, master, color=T.ACCENT, **kw):
        super().__init__(master, height=8, bg=T.CARD, highlightthickness=0, **kw)
        self.color, self.v = color, 0.0

    def __setitem__(self, key, v):
        if key == "value":
            self.v = min(max(float(v), 0.0), 1.0)
            self.delete("all")
            self.create_rectangle(0, 0, self.winfo_width() * self.v, 10, fill=self.color, outline="")


class ScrollFrame(tk.Frame):
    """Contenedor con desplazamiento vertical (rueda del ratón) y barra fina."""

    def __init__(self, master, bg=T.PANEL, **kw):
        super().__init__(master, bg=bg, **kw)
        self.cv = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.cv.yview)
        self.inner = tk.Frame(self.cv, bg=bg)
        self.inner.bind("<Configure>", lambda e: self.cv.configure(scrollregion=self.cv.bbox("all")))
        self.win = self.cv.create_window((0, 0), window=self.inner, anchor="nw")
        self.cv.bind("<Configure>", lambda e: self.cv.itemconfig(self.win, width=e.width))
        self.cv.configure(yscrollcommand=self.sb.set)
        self.cv.pack(side="left", fill="both", expand=True)
        self.sb.pack(side="right", fill="y")
        for w in (self.cv, self.inner):
            w.bind("<Enter>", self._bind_wheel)
            w.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _e=None):
        self.cv.bind_all("<MouseWheel>", self._wheel)
        self.cv.bind_all("<Button-4>", self._wheel)
        self.cv.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _e=None):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.cv.unbind_all(ev)

    def _wheel(self, e):
        if isinstance(e.widget, tk.Canvas) and getattr(e.widget, "cursor_is_slider", False):
            return
        d = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
        self.cv.yview_scroll(d, "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.cv.yview_moveto(0)


def section(master, text):
    f = tk.Frame(master, bg=master.cget("bg"))
    tk.Label(f, text=text.upper(), font=(T.FAMILY, T.FONT_SMALL[1] - 1, "bold"), fg=T.FAINT, bg=f.cget("bg")).pack(side="left")
    tk.Frame(f, bg=T.BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0), pady=1)
    return f


class EffectPicker(tk.Toplevel):
    """Ventana de búsqueda para agregar efectos (escribe para filtrar)."""

    def __init__(self, master, categories, on_pick, descriptions=None):
        super().__init__(master)
        self.on_pick, self.cats, self.desc = on_pick, categories, descriptions or {}
        self.title("Agregar efecto")
        self.configure(bg=T.PANEL)
        self.transient(master)
        self.geometry(f"380x520+{master.winfo_rootx() + master.winfo_width() - 420}+{master.winfo_rooty() + 60}")
        self.var = tk.StringVar()
        ent = ttk.Entry(self, textvariable=self.var, font=T.FONT)
        ent.pack(fill="x", padx=12, pady=(12, 8))
        ent.focus_set()
        self.lb = tk.Listbox(self, bg=T.PANEL, fg=T.TEXT, selectbackground=T.ACCENT_DIM, selectforeground=T.TEXT,
                             activestyle="none", highlightthickness=0, borderwidth=0, font=T.FONT, exportselection=False)
        self.lb.pack(fill="both", expand=True, padx=12)
        self.hint = tk.Label(self, text="", font=T.FONT_SMALL, fg=T.MUTED, bg=T.PANEL, wraplength=350, justify="left", anchor="w")
        self.hint.pack(fill="x", padx=12, pady=8)
        self.items = []
        self.var.trace_add("write", lambda *_: self._fill())
        self._fill()
        ent.bind("<Down>", lambda e: self._move(1))
        ent.bind("<Up>", lambda e: self._move(-1))
        ent.bind("<Return>", lambda e: self._pick())
        self.bind("<Escape>", lambda e: self.destroy())
        self.lb.bind("<Double-Button-1>", lambda e: self._pick())
        self.lb.bind("<<ListboxSelect>>", lambda e: self._hint())

    def _fill(self):
        q = self.var.get().strip().lower()
        self.lb.delete(0, "end")
        self.items = []
        for cat, classes in self.cats.items():
            sel = [c for c in classes if not q or q in c.name.lower() or q in cat.lower()]
            if not sel:
                continue
            self.lb.insert("end", f"  {cat.upper()}")
            self.lb.itemconfig("end", fg=T.FAINT, selectbackground=T.PANEL, selectforeground=T.FAINT)
            self.items.append(None)
            for c in sel:
                self.lb.insert("end", f"    {c.name}")
                self.items.append(c)
        first = next((i for i, c in enumerate(self.items) if c), None)
        if first is not None:
            self.lb.selection_set(first)
            self.lb.see(first)
        self._hint()

    def _move(self, d):
        cur = self.lb.curselection()
        i = (cur[0] if cur else -1) + d
        while 0 <= i < len(self.items) and self.items[i] is None:
            i += d
        if 0 <= i < len(self.items):
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(i)
            self.lb.see(i)
            self._hint()
        return "break"

    def _hint(self):
        cur = self.lb.curselection()
        c = self.items[cur[0]] if cur and cur[0] < len(self.items) else None
        self.hint.configure(text=(getattr(c, "description", "") or self.desc.get(c.name, "")) if c else "")

    def _pick(self):
        cur = self.lb.curselection()
        if cur and self.items[cur[0]]:
            cls = self.items[cur[0]]
            self.destroy()
            self.on_pick(cls)
