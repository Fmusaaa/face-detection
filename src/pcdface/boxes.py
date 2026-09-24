"""Kotak pembatas (x, y, w, h) dan operasi dasarnya."""

from __future__ import annotations

from typing import Iterable, Sequence

Box = tuple[int, int, int, int]  # (x, y, w, h) dalam piksel


def to_box(values: Sequence[float]) -> Box:
    """Ubah daftar 4 angka menjadi Box bilangan bulat."""
    if len(values) != 4:
        raise ValueError(f"kotak harus 4 angka [x, y, w, h], dapat {list(values)}")
    x, y, w, h = (int(round(float(v))) for v in values)
    return (x, y, w, h)


def clip_box(box: Box, width: int, height: int) -> Box | None:
    """Potong kotak ke batas citra. `None` bila tidak ada sisa di dalam citra."""
    x, y, w, h = box
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, width), min(y + h, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def center_x(box: Box) -> float:
    return box[0] + box[2] / 2.0


def sort_left_to_right(boxes: Iterable[Box]) -> list[Box]:
    """Urutkan kotak dari kiri ke kanan berdasarkan titik tengah horizontal."""
    return sorted(boxes, key=center_x)
