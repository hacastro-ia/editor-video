# Historial de cambios

Todos los cambios importantes de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el versionado [SemVer](https://semver.org/lang/es/).

## Cómo publicar una versión

1. Escribe los cambios nuevos en la sección **[No publicado]**.
2. Al publicar, renómbrala con la versión y la fecha: `## [1.1] - 2026-11-15`.
3. Ejecuta `.\publicar_release.ps1 -Version 1.1` — el script lee esa sección del CHANGELOG y la usa como notas del Release.
4. Deja **[No publicado]** vacío con sus categorías para la siguiente versión.

Categorías: `Añadido`, `Cambiado`, `Obsoleto`, `Eliminado`, `Corregido`, `Seguridad`.

## [No publicado]

### Añadido

- (nada todavía)

### Cambiado

### Corregido

### Eliminado

## [1.0] - 2026-10-28

Primera versión pública (beta).

### Añadido

- Cadena de **48 efectos apilables** con controles por parámetro (intensidad, mezcla, velocidad, etc.).
- Modulación reactiva al audio: beat, graves, medios, agudos, volumen y micrófono.
- Efectos de imagen: color, neón, glitch RGB, feedback, bloom, CRT/VHS, datamosh, psicodélicos, caleidoscopios y plasma generativo.
- Cámaras en vivo (hasta 1080p) y ventana de proyección en un monitor externo.
- Datamosh simulado y **real** mediante FFmpeg.
- Detección de objetos y personas con **YOLO** (recuadros, esqueleto, siluetas y seguimiento con IDs estables).
- **Censura facial** con modo automático y foto de referencia (YuNet / SFace).
- **Biblioteca de imágenes** por clase para reemplazar objetos detectados.
- Exportación a MP4, presets, Auto-VJ y guion de cinta.
- Distribución en dos formatos: **instalador** (Inno Setup, con desinstalador) y **portable**.

### Corregido

- Los modelos de IA y la biblioteca se guardan ahora en `%LOCALAPPDATA%\EditorVideo`, así funcionan aunque el programa esté instalado en `Program Files` (antes la descarga fallaba con `curl 23`).
- Incluida la extensión C++ de torchvision (`_C_stable.pyd`), que provocaba `operator torchvision::nms does not exist` al usar YOLO.
- La carpeta `biblioteca` ya no se crea en el directorio temporal `_MEIPASS`: las imágenes añadidas se conservan al cerrar el programa.
- El mensaje de ayuda del efecto de biblioteca indica la ruta real del botón: *Salida → Herramientas → «Biblioteca de imágenes…»*.

## Enlaces

[No publicado]: https://github.com/hacastro-ia/editor-video/compare/v1.0...HEAD
[1.0]: https://github.com/hacastro-ia/editor-video/releases/tag/v1.0
