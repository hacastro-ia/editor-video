"""Descarga una colección de imágenes de ejemplo desde Wikimedia Commons para la biblioteca.

Uso:
    python descargar_biblioteca.py                      # clases comunes, 12 imágenes por clase
    python descargar_biblioteca.py dog cat --n 20       # solo esas clases
    python descargar_biblioteca.py --carpeta D:/mis_imagenes

Cada clase va a su subcarpeta (biblioteca/dog, biblioteca/cat, …), con los nombres de YOLO en inglés.
Se guarda un archivo CREDITOS.txt con autor, licencia y enlace de cada imagen: las imágenes de Commons tienen
licencias libres, pero muchas exigen atribución. Si vas a publicar tus videos, revisa ese archivo.
Revisa también las carpetas: la búsqueda automática a veces trae imágenes que no son lo que esperabas
(bórralas sin problema o añade las tuyas).
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "EditorVideoPython/1.0 (proyecto personal de edicion de video)"

# clase de YOLO -> término de búsqueda (Commons funciona mejor en inglés)
CLASES = {
    "person": "people portrait photograph", "dog": "dog", "cat": "cat", "car": "car automobile",
    "bicycle": "bicycle", "motorcycle": "motorcycle", "bus": "city bus", "bird": "bird", "horse": "horse",
    "chair": "chair furniture", "bottle": "bottle", "cup": "coffee cup", "cell phone": "smartphone",
    "laptop": "laptop computer", "book": "book", "potted plant": "houseplant", "clock": "wall clock",
    "tv": "television set", "pizza": "pizza", "apple": "apple fruit", "backpack": "backpack",
    "umbrella": "umbrella", "bench": "park bench", "sheep": "sheep", "cow": "cow",
}


def _get(url, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as e:                      # red caída, límite de peticiones…
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def buscar(termino, n):
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"{termino} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": str(min(50, n * 4)),
              "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "640"}
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params)))
    pages = data.get("query", {}).get("pages", {})
    out = []
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if ii.get("width", 0) < 400 or ii.get("height", 0) < 300:
            continue
        meta = ii.get("extmetadata", {})
        out.append({
            "url": ii.get("thumburl") or ii.get("url"),
            "titulo": p.get("title", ""),
            "licencia": meta.get("LicenseShortName", {}).get("value", "?"),
            "autor": re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "?")).strip()[:120],
            "pagina": ii.get("descriptionurl", ""),
        })
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clases", nargs="*", help="clases de YOLO en inglés (por defecto, las comunes)")
    ap.add_argument("--n", type=int, default=12, help="imágenes por clase (12)")
    ap.add_argument("--carpeta", default=str(Path(__file__).resolve().parent / "biblioteca"))
    args = ap.parse_args()

    raiz = Path(args.carpeta)
    raiz.mkdir(parents=True, exist_ok=True)
    clases = args.clases or list(CLASES)
    creditos = raiz / "CREDITOS.txt"
    total = 0
    for c in clases:
        termino = CLASES.get(c, c)
        destino = raiz / c
        destino.mkdir(exist_ok=True)
        print(f"[{c}] buscando «{termino}»…", flush=True)
        try:
            items = buscar(termino, args.n)
        except Exception as e:
            print(f"   no se pudo buscar: {e}")
            continue
        ok = 0
        for k, it in enumerate(items, 1):
            ext = ".png" if it["url"].lower().endswith(".png") else ".jpg"
            ruta = destino / f"{c.replace(' ', '_')}_{k:02d}{ext}"
            if ruta.exists():
                ok += 1
                continue
            try:
                ruta.write_bytes(_get(it["url"]))
            except Exception as e:
                print(f"   falló una imagen: {e}")
                continue
            ok += 1
            with open(creditos, "a", encoding="utf-8") as f:
                f.write(f"{ruta.relative_to(raiz)} | {it['titulo']} | {it['autor']} | {it['licencia']} | {it['pagina']}\n")
            time.sleep(0.4)                             # cortesía con el servidor
        total += ok
        print(f"   {ok} imágenes en {destino}")
    print(f"\nListo: {total} imágenes en {raiz}\nCréditos y licencias: {creditos}")


if __name__ == "__main__":
    sys.exit(main())
