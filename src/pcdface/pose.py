"""Nama pose → sumbu dan sudut nominal, serta acuan ekspresi dan oklusi (PRD §6.4, E6).

Ekspresi (`netral`, `senyum`, `marah`, …) dan oklusi (`masker`, `tangan`, …)
hanya **kondisi perekaman** yang diperagakan peserta; sistem tidak pernah
menebak ekspresi, emosi, atau benda yang dipakai.

Nama pose di config dan metadata berbentuk `<arah><sudut>`:

- `depan`                         — acuan, 0°
- `kiri30`, `kanan60`, `kiri90`   — menoleh (yaw); kiri/kanan menurut **peserta**
- `menunduk30`, `mendongak30`     — mengangguk (pitch)
- `miringkiri30`, `miringkanan30` — memiringkan kepala ke bahu (roll)

Sudutnya nominal: diarahkan dengan penanda di lantai/dinding, bukan diukur
dengan alat, jadi galat ±10° wajar dan dibahas sebagai keterbatasan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REFERENCE_POSE = "depan"
REFERENCE_EXPRESSION = "netral"
REFERENCE_OCCLUSION = "tanpa"


AXES = {
    "kiri": "menoleh",
    "kanan": "menoleh",
    "menunduk": "mengangguk",
    "mendongak": "mengangguk",
    "miringkiri": "miring",
    "miringkanan": "miring",
}
AXIS_ORDER = ("depan", "menoleh", "mengangguk", "miring")

_PATTERN = re.compile(r"^(depan|kiri|kanan|menunduk|mendongak|miringkiri|miringkanan)(\d{1,2})?$")


@dataclass(frozen=True)
class PoseInfo:
    name: str
    direction: str      # depan, kiri, kanan, menunduk, ...
    axis: str           # depan, menoleh, mengangguk, miring
    angle: int          # derajat, 0 untuk depan

    def describe(self) -> str:
        """Instruksi singkat untuk peserta, ditampilkan saat merekam."""
        if self.axis == "depan":
            return "tatap lensa kamera"
        if self.axis == "menoleh":
            return f"menoleh ke {self.direction.upper()} (kiri/kanan peserta) {self.angle} derajat, bahu tetap"
        if self.axis == "mengangguk":
            return f"{self.direction.upper()} {self.angle} derajat"
        side = "KIRI" if self.direction == "miringkiri" else "KANAN"
        return f"miringkan kepala ke bahu {side} {self.angle} derajat"


def parse_pose(name: str) -> PoseInfo:
    """Uraikan nama pose. ValueError bila bentuknya tidak dikenal."""
    match = _PATTERN.match(name or "")
    if not match:
        raise ValueError(f"nama pose '{name}' tidak dikenal; contoh: depan, kiri30, menunduk30, miringkiri30")
    direction, angle = match.group(1), match.group(2)
    if direction == REFERENCE_POSE:
        if angle:
            raise ValueError(f"pose '{name}': 'depan' tidak memakai sudut")
        return PoseInfo(name, direction, "depan", 0)
    if not angle or not 0 < int(angle) <= 90:
        raise ValueError(f"pose '{name}': sudut harus 1–90 derajat, mis. {direction}30")
    return PoseInfo(name, direction, AXES[direction], int(angle))
