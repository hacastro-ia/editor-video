"""Análisis de audio: graves, medios, agudos, volumen, beat y espectro.

Solo se usa NumPy + FFmpeg (para decodificar). El audio original nunca se modifica:
las señales sirven únicamente para controlar los efectos visuales.
"""
from __future__ import annotations

import subprocess
from collections import deque

import numpy as np

from core.params import N_SPEC, zero_audio
from core.utils import ffmpeg_exe, popen_kwargs

try:  # reproducción / micrófono (opcional)
    import sounddevice as sd
except Exception:  # PortAudio ausente o librería no instalada
    sd = None

SR = 44100
N_FFT = 2048
HOP = 1024
COLS_PER_SEC = SR / HOP

_FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SR)


def _bins(lo, hi):
    return int(np.searchsorted(_FREQS, lo)), int(np.searchsorted(_FREQS, hi))


BASS = _bins(20, 250)
MID = _bins(250, 2000)
TREB = _bins(2000, 8000)
F_MAX = _bins(0, 4000)[1]                      # el flujo espectral usa hasta 4 kHz
_EDGES = np.searchsorted(_FREQS, np.geomspace(40, 12000, N_SPEC + 1))
_WIN = np.hanning(N_FFT).astype(np.float32)


def spectral_features(mag, prev_logm):
    """mag: (columnas, bins). Devuelve dict de arrays y el último log-espectro."""
    bass = mag[:, BASS[0]:BASS[1]].mean(axis=1)
    mid = mag[:, MID[0]:MID[1]].mean(axis=1)
    treb = mag[:, TREB[0]:TREB[1]].mean(axis=1)
    spec = np.stack([mag[:, a:max(b, a + 1)].mean(axis=1)
                     for a, b in zip(_EDGES[:-1], _EDGES[1:])], axis=1)
    logm = np.log1p(mag[:, :F_MAX] * 30.0)
    d = np.diff(np.vstack([prev_logm[None, :], logm]), axis=0)
    flux = np.clip(d, 0, None).sum(axis=1)
    return {"bass": bass, "mid": mid, "treb": treb, "spec": spec, "flux": flux}, logm[-1]


class BeatTracker:
    """Detector de beats incremental: pulso 1.0 que decae rápido."""

    def __init__(self, sensitivity=1.45):
        self.sens = sensitivity
        self.hist = deque(maxlen=43)
        self.peak = 1e-6
        self.since = 99
        self.cur = 0.0

    def push(self, flux):
        self.peak = max(self.peak * 0.998, flux)
        self.since += 1
        mean = (sum(self.hist) / len(self.hist)) if self.hist else 0.0
        if self.hist and flux > self.sens * mean + 1e-6 and flux > 0.3 * self.peak and self.since >= 8:
            self.cur = 1.0
            self.since = 0
        else:
            self.cur *= 0.85
        self.hist.append(flux)
        return self.cur


def beat_env_from_flux(flux, sens=1.45):
    tr = BeatTracker(sens)
    return np.array([tr.push(float(f)) for f in flux], np.float32)


def _follow(x, release):
    """Seguidor de envolvente: ataque instantáneo, caída suave."""
    out = np.empty_like(x)
    cur = 0.0
    for i, v in enumerate(x):
        cur = v if v > cur else cur * release + v * (1.0 - release)
        out[i] = cur
    return out


class AudioAnalysis:
    """Análisis completo (precalculado) del audio de un archivo de video."""

    def __init__(self):
        self.pcm = None                 # (N, 2) float32 para reproducir
        self.duration = 0.0
        self.flux = None
        self.arrays = None
        self.beat = None
        self.sens = 1.45
        self.muted = False
        self.play_error = ""

    # -- carga -------------------------------------------------------------
    def load(self, path):
        exe = ffmpeg_exe()
        if not exe:
            raise RuntimeError("No se encontró FFmpeg (instala imageio-ffmpeg o FFmpeg).")
        cmd = [exe, "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
        proc = subprocess.run(cmd, capture_output=True, **popen_kwargs())
        raw = proc.stdout
        if not raw:
            raise RuntimeError("El video no tiene pista de audio.")
        arr = np.frombuffer(raw, dtype=np.float32)
        arr = arr[: len(arr) // 2 * 2].reshape(-1, 2)
        self.pcm = np.ascontiguousarray(arr)
        self.duration = len(arr) / SR
        self._analyze(arr.mean(axis=1))

    def _analyze(self, mono):
        if len(mono) < N_FFT:
            mono = np.pad(mono, (0, N_FFT - len(mono)))
        view = np.lib.stride_tricks.sliding_window_view(mono, N_FFT)[::HOP]
        n = view.shape[0]
        parts = {k: [] for k in ("bass", "mid", "treb", "flux", "rms")}
        specs = []
        prev = np.zeros(F_MAX)
        for i in range(0, n, 1500):
            chunk = np.asarray(view[i:i + 1500], dtype=np.float32)
            rms = np.sqrt((chunk ** 2).mean(axis=1))
            mag = np.abs(np.fft.rfft(chunk * _WIN, axis=1)) / (N_FFT / 2)
            feats, prev = spectral_features(mag, prev)
            for k in ("bass", "mid", "treb", "flux"):
                parts[k].append(feats[k])
            parts["rms"].append(rms)
            specs.append(feats["spec"])
        arr = {k: np.concatenate(v).astype(np.float32) for k, v in parts.items()}
        spec = np.concatenate(specs).astype(np.float32)

        def norm(x, release):
            p = max(float(np.percentile(x, 99)), 1e-6)
            return _follow(np.clip(x / p, 0, 1), release)

        self.arrays = {
            "graves": norm(arr["bass"], 0.80),
            "medios": norm(arr["mid"], 0.75),
            "agudos": norm(arr["treb"], 0.70),
            "volumen": norm(arr["rms"], 0.80),
        }
        sp = max(float(np.percentile(spec, 99)), 1e-6)
        self.spec = np.clip(spec / sp, 0, 1) ** 0.6
        self.flux = arr["flux"]
        self.beat = beat_env_from_flux(self.flux, self.sens)

    def set_sensitivity(self, sens):
        self.sens = float(sens)
        if self.flux is not None:
            self.beat = beat_env_from_flux(self.flux, self.sens)

    # -- consulta ----------------------------------------------------------
    def features_at(self, t):
        if self.arrays is None:
            return zero_audio()
        n = len(self.beat)
        i = int((t * SR - N_FFT / 2) / HOP + 0.5)
        i = min(max(i, 0), n - 1)
        return {
            "graves": float(self.arrays["graves"][i]),
            "medios": float(self.arrays["medios"][i]),
            "agudos": float(self.arrays["agudos"][i]),
            "volumen": float(self.arrays["volumen"][i]),
            "beat": float(self.beat[i]),
            "spectrum": self.spec[i],
        }

    # -- reproducción (solo vista previa) ---------------------------------
    def play(self, t):
        if self.pcm is None or sd is None or self.muted:
            return
        try:
            sd.play(self.pcm[int(max(t, 0.0) * SR):], SR, blocking=False)
        except Exception as e:
            self.play_error = str(e)

    def stop(self):
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass


class LiveAudio:
    """Análisis en tiempo real del micrófono (o de la mezcla estéreo si el sistema lo permite)."""

    def __init__(self):
        self.stream = None
        self.error = ""
        self.buf = np.zeros(N_FFT, np.float32)
        self.prev = np.zeros(F_MAX)
        self.peak = {"bass": 1e-3, "mid": 1e-3, "treb": 1e-3, "rms": 1e-3, "spec": 1e-3}
        self.env = {"graves": 0.0, "medios": 0.0, "agudos": 0.0, "volumen": 0.0}
        self.tracker = BeatTracker()
        self.latest = zero_audio()

    def set_sensitivity(self, sens):
        self.tracker.sens = float(sens)

    def start(self):
        if sd is None:
            self.error = "sounddevice no está disponible (pip install sounddevice)."
            return False
        try:
            self.stream = sd.InputStream(samplerate=SR, channels=1, blocksize=HOP,
                                         dtype="float32", callback=self._callback)
            self.stream.start()
            return True
        except Exception as e:
            self.error = f"No se pudo abrir el micrófono: {e}"
            self.stream = None
            return False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _norm(self, key, v):
        self.peak[key] = max(self.peak[key] * 0.999, v, 2e-3)
        return min(v / self.peak[key], 1.0)

    def _callback(self, indata, frames, time_info, status):
        x = indata[:, 0]
        if len(x) < HOP:
            x = np.pad(x, (0, HOP - len(x)))
        x = x[:HOP]
        self.buf = np.concatenate([self.buf[HOP:], x])
        mag = (np.abs(np.fft.rfft(self.buf * _WIN)) / (N_FFT / 2))[None, :]
        feats, self.prev = spectral_features(mag, self.prev)
        raw = {"graves": self._norm("bass", float(feats["bass"][0])),
               "medios": self._norm("mid", float(feats["mid"][0])),
               "agudos": self._norm("treb", float(feats["treb"][0])),
               "volumen": self._norm("rms", float(np.sqrt((self.buf ** 2).mean())))}
        rel = {"graves": 0.80, "medios": 0.75, "agudos": 0.70, "volumen": 0.80}
        for k, v in raw.items():
            cur = self.env[k]
            self.env[k] = v if v > cur else cur * rel[k] + v * (1 - rel[k])
        spec = feats["spec"][0]
        self.peak["spec"] = max(self.peak["spec"] * 0.999, float(spec.max()), 2e-3)
        beat = self.tracker.push(float(feats["flux"][0]))
        self.latest = {**self.env, "beat": beat,
                       "spectrum": (np.clip(spec / self.peak["spec"], 0, 1) ** 0.6).astype(np.float32)}

    def features_at(self, t):
        return self.latest
