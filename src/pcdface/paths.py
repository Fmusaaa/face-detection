"""Path bawaan proyek.

Semua path dihitung dari PROJECT_ROOT, bukan dari folder tempat perintah
dijalankan. Modul lain menerima `ProjectPaths` supaya run sintetis dan tes
bisa menunjuk ke folder sementara tanpa menyentuh `data/` asli.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# src/pcdface/paths.py -> akar proyek dua tingkat di atas paket
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

_REQUIRED_KEYS = ("raw", "metadata", "subjects", "annotations", "crops", "models", "results")
# Kunci opsional dan nilai bawaannya (relatif terhadap root) — config lama tetap terbaca
_OPTIONAL_KEYS = {"recognition": "data/recognition"}


def resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Path absolute dibiarkan, path relatif dianggap relatif terhadap `root`."""
    path = Path(path)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class ProjectPaths:
    """Lokasi seluruh berkas data dan keluaran untuk satu run."""

    raw: Path
    metadata: Path
    subjects: Path
    annotations: Path
    crops: Path
    models: Path
    results: Path
    # model LBPH terlatih + daftar label. Berisi templat biometrik — data pribadi, di-gitignore (PRD §11)
    recognition: Path = PROJECT_ROOT / "data" / "recognition"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], root: Path = PROJECT_ROOT) -> "ProjectPaths":
        missing = [key for key in _REQUIRED_KEYS if key not in mapping]
        if missing:
            raise KeyError(f"paths tidak lengkap, kurang: {', '.join(missing)}")
        values = {key: resolve(mapping[key], root) for key in _REQUIRED_KEYS}
        values.update({key: resolve(mapping.get(key, default), root) for key, default in _OPTIONAL_KEYS.items()})
        return cls(**values)

    def with_data_root(self, data_root: Path) -> "ProjectPaths":
        """Salinan dengan data di `data_root` (untuk dataset sintetis).

        Model tetap menunjuk ke folder model proyek; hasil ditaruh terpisah
        di `results/synthetic/` supaya tidak tercampur dengan hasil asli.
        """
        return ProjectPaths(
            raw=data_root / "raw",
            metadata=data_root / "metadata.csv",
            subjects=data_root / "subjects.csv",
            annotations=data_root / "annotations" / "boxes.json",
            crops=data_root / "crops",
            models=self.models,
            results=self.results / "synthetic",
            recognition=data_root / "recognition",
        )
