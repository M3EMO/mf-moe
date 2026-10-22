#!/usr/bin/env bash
# Arma el zip de un job de Colab desde el último commit (nunca desde archivos sin commitear) y le agrega COMMIT con
# su hash: cada result.json de la fase 2 guarda de qué código salió (I6 de la spec). Uso: build_zip.sh SALIDA.zip
set -eu
OUT=$(realpath -m "$1")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git -C "$ROOT" archive HEAD LICENSE pyproject.toml colab mf_moe | tar -x -C "$TMP"
git -C "$ROOT" rev-parse HEAD > "$TMP/COMMIT"
(cd "$TMP" && rm -f "$OUT.tmp" && zip -qr "$OUT.tmp" COMMIT LICENSE pyproject.toml colab/*.py mf_moe/*.py)
mv "$OUT.tmp" "$OUT"
echo "$OUT $(cat "$TMP/COMMIT")"
