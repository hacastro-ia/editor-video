"""Datamosh REAL sobre archivos de video, manipulando el flujo de bits del códec (como Datamosher Pro).

Cómo funciona:
  1. Se recodifica el video a MPEG-4 Parte 2 sin fotogramas B (flujo «m4v» crudo) con fotogramas clave (I) solo donde
     se pide.
  2. El flujo se corta en paquetes (un fotograma cada uno). Un I-frame lleva la imagen completa; un P-frame solo
     lleva movimiento y diferencias respecto al fotograma anterior.
  3. Se editan los paquetes: se QUITAN I-frames (el movimiento del video nuevo se aplica sobre la imagen vieja),
     se REPITEN P-frames (el mismo movimiento se acumula: «bloom») o se BARAJAN trozos.
  4. Se decodifica el flujo roto y se guarda como MP4. El decodificador, al faltarle la referencia, «derrite» la imagen.

Necesita FFmpeg (imageio-ffmpeg lo trae incluido).
"""
from __future__ import annotations

import os
import random
import subprocess
import tempfile

import cv2

from core.utils import ffmpeg_exe, popen_kwargs

VOP = b"\x00\x00\x01\xb6"
VOS = b"\x00\x00\x01\xb0"
MODES = ["Clásico: quitar I-frames", "Bloom: repetir P-frames", "Combinar varios videos", "Barajar trozos"]


def _run(cmd, progress=None):
    p = subprocess.run(cmd, capture_output=True, **popen_kwargs())
    if p.returncode != 0:
        raise RuntimeError("FFmpeg falló: " + p.stderr.decode(errors="ignore")[-500:])


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def encode_m4v(src, dst, fps, size, key_times=(), quality=4, start=None, duration=None):
    """Video -> flujo MPEG-4 crudo (sin B-frames). I-frames al inicio y en key_times (segundos)."""
    ff = ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", src]
    if duration:
        cmd += ["-t", f"{duration}"]
    cmd += ["-an", "-vf", f"scale={size[0]}:{size[1]},fps={fps}", "-c:v", "mpeg4", "-q:v", str(quality), "-bf", "0",
            "-g", "100000", "-sc_threshold", "0", "-pix_fmt", "yuv420p"]
    if key_times:
        cmd += ["-force_key_frames", ",".join(f"{t:.3f}" for t in sorted(key_times))]
    cmd += ["-f", "m4v", dst]
    _run(cmd)


def split_packets(data: bytes):
    """Devuelve [(bytes, tipo)] con tipo 'I', 'P', 'B' o 'S'. Los encabezados (VOS/VOL) quedan en su I-frame."""
    starts = []
    i = 0
    n = len(data)
    pending_head = None
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0 or j + 3 >= n:
            break
        code = data[j + 3]
        if code == 0xB0 and pending_head is None:
            pending_head = j
        elif code == 0xB6:
            starts.append(pending_head if pending_head is not None else j)
            pending_head = None
        i = j + 3
    packets = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else n
        pkt = data[s:e]
        v = pkt.find(VOP)
        kind = "IPBS"[pkt[v + 4] >> 6] if v >= 0 else "P"
        packets.append((pkt, kind))
    return packets


def decode_m4v(m4v_path, out_path, fps, audio_src=None, audio_start=None):
    ff = ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "m4v", "-framerate", f"{fps}", "-i", m4v_path]
    if audio_src:
        if audio_start:
            cmd += ["-ss", f"{audio_start}"]
        cmd += ["-i", audio_src]
    cmd += ["-map", "0:v:0"]
    if audio_src:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    _run(cmd)


def datamosh_file(sources, out_path, mode=MODES[0], size=None, gop_seconds=2.0, remove="todos", every=1,
                  range_s=(0.0, 1e9), repeat_times=(), repeat_count=24, repeat_len=1, seed=1, quality=4,
                  keep_audio=True, log=None):
    """Genera un video con datamosh real. Devuelve un mensaje con lo que hizo."""
    say = log or (lambda m: None)
    if not ffmpeg_exe():
        raise RuntimeError("No se encontró FFmpeg (pip install imageio-ffmpeg).")
    fps, _, w0, h0 = video_info(sources[0])
    w, h = size or (w0 - w0 % 2, h0 - h0 % 2)
    w, h = w - w % 2, h - h % 2
    rng = random.Random(seed)
    tmp = tempfile.mkdtemp(prefix="mosh_")
    try:
        packets = []
        if mode == MODES[2]:                                   # combinar: se quita el I-frame inicial de cada video siguiente
            for k, src in enumerate(sources):
                say(f"Codificando {os.path.basename(src)}…")
                m4v = os.path.join(tmp, f"c{k}.m4v")
                encode_m4v(src, m4v, fps, (w, h), quality=quality)
                with open(m4v, "rb") as f:
                    pk = split_packets(f.read())
                packets += pk if k == 0 else [p for i, p in enumerate(pk) if not (i == 0 and p[1] == "I")]
            msg = f"{len(sources)} videos combinados quitando el I-frame de cada transición."
        else:
            dur = video_info(sources[0])[1] / fps
            keys = []
            if mode in (MODES[0], MODES[3]):
                t = gop_seconds
                while t < dur:
                    keys.append(t)
                    t += gop_seconds
            say("Codificando a MPEG-4…")
            m4v = os.path.join(tmp, "a.m4v")
            encode_m4v(sources[0], m4v, fps, (w, h), key_times=keys, quality=quality)
            with open(m4v, "rb") as f:
                pk = split_packets(f.read())
            if mode == MODES[0]:                               # clásico
                out, ni = [], 0
                for i, (data, kind) in enumerate(pk):
                    t = i / fps
                    if kind == "I" and i > 0 and range_s[0] <= t <= range_s[1]:
                        ni += 1
                        if remove == "todos" or (ni % max(1, every) == 0):
                            continue                           # se elimina este I-frame
                    out.append((data, kind))
                packets = out
                msg = f"Se eliminaron {len(pk) - len(out)} I-frames de {sum(1 for _, k in pk if k == 'I')}."
            elif mode == MODES[1]:                             # bloom: repetir P-frames
                marks = sorted({int(t * fps) for t in repeat_times if t * fps < len(pk)})
                if not marks:
                    marks = [len(pk) // 3]
                out, i = [], 0
                while i < len(pk):
                    if i in marks and pk[i][1] == "P":
                        block = pk[i:i + max(1, repeat_len)]
                        out += block * max(1, repeat_count)
                        i += len(block)
                    else:
                        out.append(pk[i])
                        i += 1
                packets = out
                msg = f"Se repitieron {len(marks)} bloque(s) de P-frames ×{repeat_count}."
            else:                                              # barajar
                chunks, cur = [], []
                for data, kind in pk:
                    if kind == "I" and cur:
                        chunks.append(cur)
                        cur = []
                    cur.append((data, kind))
                chunks.append(cur)
                first, rest = chunks[0], chunks[1:]
                rng.shuffle(rest)
                packets = list(first)
                for c in rest:                                 # sin I-frame: cada trozo aplica su movimiento sobre el anterior
                    packets += [p for i, p in enumerate(c) if not (i == 0 and p[1] == "I")]
                msg = f"Se barajaron {len(rest)} trozos de {gop_seconds:.1f} s."
        edited = os.path.join(tmp, "edit.m4v")
        with open(edited, "wb") as f:
            for data, _ in packets:
                f.write(data)
        say("Decodificando el flujo roto…")
        decode_m4v(edited, out_path, fps, audio_src=sources[0] if keep_audio and mode != MODES[2] else None)
        return f"Listo: {len(packets)} fotogramas. {msg}"
    finally:
        for fn in os.listdir(tmp):
            try:
                os.remove(os.path.join(tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass
