"""Tulis tabel sebagai `.csv` (presisi penuh, titik desimal) dan `.md` (siap tempel).

Di `.md`, kolom `x` yang punya pasangan `x_low` dan `x_high` digabung menjadi
`0,931 [0,852; 0,970]`, dan angka memakai koma desimal sesuai kaidah
penulisan bahasa Indonesia. CSV tetap memakai titik agar mudah diolah ulang.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

DECIMALS = 3


def _fmt_number(value: Any, decimals: int = DECIMALS) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "ya" if value else "tidak"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "–"
        if value.is_integer() and abs(value) >= 1000:
            return f"{int(value)}"
        return f"{value:.{decimals}f}".replace(".", ",")
    return str(value)


def _fmt_interval(value: Any, low: Any, high: Any, decimals: int) -> str:
    if pd.isna(value):
        return "–"
    text = _fmt_number(float(value), decimals)
    if pd.isna(low) or pd.isna(high):
        return text
    return f"{text} [{_fmt_number(float(low), decimals)}; {_fmt_number(float(high), decimals)}]"


def display_frame(df: pd.DataFrame, decimals: int = DECIMALS) -> pd.DataFrame:
    """Gabungkan kolom interval dan format angka untuk tampilan Markdown."""
    shown: dict[str, list[str]] = {}
    skip = set()
    for column in df.columns:
        if column in skip:
            continue
        low, high = f"{column}_low", f"{column}_high"
        if low in df.columns and high in df.columns:
            skip.update((low, high))
            shown[column] = [_fmt_interval(v, lo, hi, decimals) for v, lo, hi in zip(df[column], df[low], df[high])]
        else:
            shown[column] = [_fmt_number(v.item() if hasattr(v, "item") else v, decimals) for v in df[column]]
    return pd.DataFrame(shown)


def to_markdown(df: pd.DataFrame, decimals: int = DECIMALS) -> str:
    frame = display_frame(df, decimals)
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, title: str = "", notes: str = "") -> Path:
    """Tulis `<stem>.csv` dan `<stem>.md`. Kembalikan path CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    parts = []
    if title:
        parts.append(f"**{title}**\n")
    parts.append(to_markdown(df))
    if notes:
        parts.append(f"\n{notes}")
    (out_dir / f"{stem}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    return csv_path


def read_table(out_dir: Path, stem: str) -> pd.DataFrame:
    return pd.read_csv(out_dir / f"{stem}.csv")
