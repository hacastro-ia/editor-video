"""Exportación del video editado a MP4 (con el audio original si hay FFmpeg).

Dos modos:
  · lineal: recorre el video de principio a fin;
  · con guion de cinta (plan): sigue la lista [(fotograma_de_origen, estado)] generada por core.transport,
    con pausas, rebobinados y avances. El sonido solo suena en las partes de reproducción normal.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave

import cv2
import numpy as np

from core.params import zero_audio
from core.transport import FrameFetcher
from core.utils import ffmpeg_exe, popen_kwargs, prepare_frame

SR = 44100


def build_plan_audio(pcm, plan, fps, path, fade=256):
    """Escribe un WAV con el audio que corresponde a cada fotograma del plan (silencio fuera de «play»)."""
    n_out = int(round(len(plan) * SR / fps))
    out = np.zeros((n_out, 2), np.float32)
    runs = []
    for j, (idx, state) in enumerate(plan):
        if state != "play":
            continue
        a, b = int(round(j * SR / fps)), int(round((j + 1) * SR / fps))
        s = int(round(idx * SR / fps))
        seg = pcm[s:s + (b - a)]
        if len(seg) == 0:
            continue
        out[a:a + len(seg)] = seg
        if runs and runs[-1][1] == a:
            runs[-1][1] = a + len(seg)
        else:
            runs.append([a, a + len(seg)])
    for a, b in runs:                                  # fundidos cortos: sin «clics» al cortar el sonido
        f = min(fade, (b - a) // 2)
        if f > 0:
            ramp = np.linspace(0, 1, f, dtype=np.float32)[:, None]
            out[a:a + f] *= ramp
            out[b - f:b] *= ramp[::-1]
    data = (np.clip(out, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def _start_encoder(out_path, w, h, fps, audio_input):
    """Devuelve (proceso_ffmpeg, writer_cv2, nota). audio_input: ruta de un archivo con el audio o None."""
    ff = ffmpeg_exe()
    if ff:
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        if audio_input:
            cmd += ["-i", audio_input]
        cmd += ["-map", "0:v:0"]
        if audio_input:
            cmd += ["-map", "1:a:0?"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kwargs())
        return proc, None, ""
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return None, writer, "FFmpeg no encontrado: el video se guardó SIN audio."


def _finish(proc, writer, cancelled):
    if proc is not None:
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode(errors="ignore")
        rc = proc.wait()
        if rc != 0 and not cancelled:
            raise RuntimeError(f"FFmpeg falló: {err[-400:]}")
    if writer is not None:
        writer.release()


def export_video(src_path, out_path, chain, ctx, max_width, audio=None, progress=None, cancel=None,
                 plan=None, pcm=None):
    """Procesa los fotogramas con la cadena de efectos y guarda el MP4.

    audio: análisis que alimenta las señales de los efectos (o None).
    plan / pcm: guion de cinta y audio original (AudioAnalysis) para reconstruir el sonido.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video de origen.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fetcher = FrameFetcher(cap) if plan else None
    first_raw = fetcher.get(plan[0][0]) if plan else cap.read()[1]
    if first_raw is None:
        raise RuntimeError("El video no tiene fotogramas.")
    first = prepare_frame(first_raw, max_width)
    h, w = first.shape[:2]

    chain.reset()
    if ctx.faces is not None:
        ctx.faces.clear_tracks()

    tmp_wav = None
    audio_input = None
    warn = ""
    if plan:
        if pcm is not None and getattr(pcm, "pcm", None) is not None:
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            build_plan_audio(pcm.pcm, plan, fps, tmp_wav)
            audio_input = tmp_wav
        else:
            warn = " (sin audio: no se pudo leer la pista de sonido)"
    else:
        audio_input = src_path
    proc, writer, no_ffmpeg = _start_encoder(out_path, w, h, fps, audio_input)

    n_total = len(plan) if plan else total
    i = 0
    cancelled = False
    try:
        while True:
            if plan:
                if i >= len(plan):
                    break
                idx, state = plan[i]
                raw = fetcher.get(idx)
                if raw is None:
                    break
                f = first if i == 0 else prepare_frame(raw, max_width)
                ctx.transport = state
                ctx.time = i / fps
                ctx.audio = audio.features_at(idx / fps) if (audio is not None and state == "play") else zero_audio()
            else:
                if i == 0:
                    f = first
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    f = prepare_frame(frame, max_width)
                ctx.transport = "play"
                ctx.time = i / fps
                ctx.audio = audio.features_at(ctx.time) if audio is not None else zero_audio()
            ctx.frame_index = i
            ctx.fps = fps
            out = chain.process(f, ctx)
            if out.shape[:2] != (h, w):
                out = cv2.resize(out, (w, h))
            out = np.ascontiguousarray(out, dtype=np.uint8)
            if proc is not None:
                proc.stdin.write(out.tobytes())
            else:
                writer.write(out)
            i += 1
            if progress:
                progress(i, n_total)
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
    finally:
        cap.release()
        try:
            _finish(proc, writer, cancelled)
        finally:
            if tmp_wav:
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
    if cancelled:
        try:
            os.remove(out_path)
        except OSError:
            pass
        return "Exportación cancelada."
    if no_ffmpeg:
        return no_ffmpeg
    return f"Listo: {i} fotogramas exportados{warn}."
