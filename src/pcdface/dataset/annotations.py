"""Baca/tulis `boxes.json` — kotak wajah manual (ground truth).

Format:

    {
      "version": 1,
      "annotations": {
        "jarak/S03/jarak_S03_150cm_normal_04.jpg": [[512, 180, 120, 158]],
        "kosong/kosong_07.jpg": []
      }
    }

Kunci adalah path relatif terhadap `data/raw/`. Daftar kosong berarti citra
sudah dianotasi dan memang tidak memuat wajah — berbeda dari kunci yang tidak
ada (belum dianotasi). Tidak ada medan nama atau identitas.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pcdface.boxes import Box, to_box

FORMAT_VERSION = 1


def load_annotations(path: Path) -> dict[str, list[Box]]:
    """Baca anotasi. Berkas yang belum ada dianggap kosong."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("annotations", payload) if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: format anotasi tidak dikenal")
    annotations: dict[str, list[Box]] = {}
    for key, boxes in raw.items():
        if not isinstance(boxes, list):
            raise ValueError(f"{path}: nilai untuk {key} harus daftar kotak")
        annotations[key] = [to_box(box) for box in boxes]
    return annotations


def save_annotations(path: Path, annotations: dict[str, list[Box]]) -> None:
    """Tulis anotasi secara atomik, kunci terurut agar diff mudah dibaca."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "catatan": "Ground truth deteksi wajah: hanya kotak [x, y, w, h], tanpa identitas.",
        "annotations": {key: [list(box) for box in annotations[key]] for key in sorted(annotations)},
    }
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
