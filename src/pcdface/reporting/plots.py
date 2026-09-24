"""Delapan grafik minimum PRD §10 — PNG 300 dpi + PDF, dibangun dari CSV hasil.

Semua grafik hanya membaca tabel `.csv` di folder eksperimen, sehingga
`python -m pcdface report` bisa membangunnya ulang tanpa menjalankan detektor.

Aturan tampilan:
- warna mengikuti detektor (bukan urutan), dari palet kategorikal yang sudah
  divalidasi aman buta warna; setiap detektor juga punya penanda sendiri
  sehingga tetap terbaca dalam cetak hitam-putih;
- satu sumbu-y per panel (tidak ada sumbu ganda); besaran berbeda → panel terpisah;
- pita/garis galat = interval kepercayaan dari tabel;
- label dan judul dalam bahasa Indonesia, angka memakai koma desimal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Warna mengikuti entitas. Urutan ini lolos validator (CVD ΔE ≥ 9,2 antarpasangan bersebelahan).
DETECTOR_STYLE: dict[str, tuple[str, str]] = {
    "haar": ("#2a78d6", "o"),
    "haar_eq": ("#2a78d6", "o"),
    "haar_noeq": ("#4a3aa7", "D"),
    "mp_short": ("#eb6834", "s"),
    "mp_full": ("#1baf7a", "^"),
    "mp_sparse": ("#4a3aa7", "v"),
    "ycbcr": ("#e87ba4", "P"),
}
FALLBACK = ("#52514e", "x")
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABEL = {
    "haar": "Haar", "haar_eq": "Haar (equalize)", "haar_noeq": "Haar (tanpa equalize)",
    "mp_short": "MediaPipe short", "mp_full": "MediaPipe full", "mp_sparse": "MediaPipe sparse",
    "ycbcr": "YCbCr",
}
DPI = 300


def _style(name: str) -> tuple[str, str]:
    return DETECTOR_STYLE.get(name, FALLBACK)


def _label(name: str) -> str:
    return LABEL.get(name, name)


def _comma(decimals: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v:.{decimals}f}".replace(".", ","))


def _setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.family": "sans-serif",
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _read(out_dir: Path, stem: str) -> pd.DataFrame | None:
    path = out_dir / f"{stem}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _recall_axis(ax: plt.Axes) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(_comma(1))
    ax.set_ylabel("Recall")


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------
def plot_recall_vs_distance(out_dir: Path) -> list[Path]:
    """Grafik 1: recall terhadap jarak per detektor, pita Wilson, garis 200 cm."""
    df = _read(out_dir, "e1_recall_per_jarak")
    if df is None:
        return []
    resolution = df["resolusi"].iloc[0]
    df = df[df["resolusi"] == resolution]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for name, part in df.groupby("detektor", sort=False):
        color, marker = _style(name)
        part = part.sort_values("jarak_cm")
        ax.fill_between(part["jarak_cm"], part["recall_low"], part["recall_high"], color=color, alpha=0.14, linewidth=0)
        ax.plot(part["jarak_cm"], part["recall"], color=color, marker=marker, label=_label(name))
    ax.axvline(200, color=MUTED, linewidth=1.0, linestyle="--")
    ax.text(196, 0.5, "klaim short-range ≤ 2 m", color=INK_2, fontsize=8, rotation=90, ha="right", va="center")
    fig.text(0.01, -0.03, "Klaim full-range (≤ 5 m) berada di luar rentang data "
             f"(maks. {int(df['jarak_cm'].max())} cm): hanya bisa dinilai 'konsisten dengan'.",
             color=INK_2, fontsize=8, ha="left")
    ax.set_xlabel("Jarak kamera–wajah (cm)")
    _recall_axis(ax)
    ax.set_xticks(sorted(df["jarak_cm"].unique()))
    ax.set_title(f"Recall terhadap jarak ({resolution.replace('x', '×')}), pita = Wilson 95%", loc="left")
    ax.legend(loc="lower left")
    return _save(fig, out_dir, "grafik1_recall_jarak")


def plot_width_loglog(out_dir: Path) -> list[Path]:
    """Grafik 2: lebar wajah (px) terhadap jarak, log-log, garis regresi + persamaan."""
    widths, fits = _read(out_dir, "e1_recall_per_jarak"), _read(out_dir, "e1_loglog")
    if widths is None or fits is None:
        return []
    resolution = fits["resolusi"].iloc[0]
    fit = fits[fits["resolusi"] == resolution].iloc[0]
    part = widths[(widths["resolusi"] == resolution) & (widths["detektor"] == widths["detektor"].iloc[0])]
    part = part.sort_values("jarak_cm")
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.errorbar(part["jarak_cm"], part["lebar_px"], yerr=part["lebar_px_sb"], fmt="o", color=INK_2,
                ecolor=MUTED, elinewidth=1, capsize=3, markersize=6, label="rerata ± sb (kotak manual)")
    z = np.linspace(part["jarak_cm"].min() * 0.9, part["jarak_cm"].max() * 1.1, 100)
    ax.plot(z, np.exp(fit["intersep"]) * z ** fit["kemiringan"], color="#2a78d6", label="regresi log-log")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted(part["jarak_cm"].unique())
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    low, high = part["lebar_px"].min() * 0.8, part["lebar_px"].max() * 1.2
    y_ticks = [v for v in (10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600) if low <= v <= high]
    ax.set_yticks(y_ticks, [str(v) for v in y_ticks])
    ax.set_ylim(low, high)
    slope = f"{fit['kemiringan']:.3f}".replace(".", ",")
    r2 = f"{fit['r2']:.3f}".replace(".", ",")
    ax.text(0.97, 0.93, f"w = {np.exp(fit['intersep']):.0f} · Z^{slope}\nR² = {r2}", transform=ax.transAxes,
            ha="right", va="top", color=INK, fontsize=9)
    ax.set_xlabel("Jarak Z (cm, skala log)")
    ax.set_ylabel("Lebar wajah (px, skala log)")
    ax.set_title(f"Lebar wajah terhadap jarak ({resolution.replace('x', '×')}) — kemiringan ≈ −1 bila w ∝ 1/Z",
                 loc="left")
    ax.legend(loc="lower left")
    return _save(fig, out_dir, "grafik2_lebar_loglog")


def _pooled_bins(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["detektor", "bin_bawah", "bin_atas"], sort=False)[["wajah", "TP"]].sum().reset_index()
    grouped = grouped[grouped["wajah"] > 0].copy()
    grouped["recall"] = grouped["TP"] / grouped["wajah"]
    grouped["tengah"] = np.where(grouped["bin_atas"] > grouped["bin_bawah"] * 4,
                                 grouped["bin_bawah"] * 1.25, (grouped["bin_bawah"] + grouped["bin_atas"]) / 2)
    return grouped


def plot_recall_vs_size(out_dir: Path) -> list[Path]:
    """Grafik 3: recall terhadap lebar wajah — piksel (kiri) dan proporsi frame (kanan)."""
    px, ratio = _read(out_dir, "e1_ukuran_px"), _read(out_dir, "e1_ukuran_proporsi")
    if px is None or ratio is None:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, df, xlabel, factor in ((axes[0], px, "Lebar wajah (px)", 1.0),
                                   (axes[1], ratio, "Lebar wajah (% lebar frame)", 100.0)):
        for name, part in _pooled_bins(df).groupby("detektor", sort=False):
            color, marker = _style(name)
            ax.plot(part["tengah"] * factor, part["recall"], color=color, marker=marker, label=_label(name))
        ax.set_xlabel(xlabel + " — kedua resolusi digabung")
        ax.xaxis.set_major_formatter(_comma(0))
    _recall_axis(axes[0])
    axes[1].set_ylabel("")
    axes[0].set_title("Batas absolut: piksel", loc="left")
    axes[1].set_title("Batas relatif: proporsi frame", loc="left")
    axes[0].legend(loc="lower right")
    fig.suptitle("Recall terhadap ukuran wajah", x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, "grafik3_recall_ukuran")


def plot_resolution_effect(out_dir: Path) -> list[Path]:
    """Grafik 4: recall terhadap jarak, resolusi besar vs kecil, satu panel per detektor (H2)."""
    df = _read(out_dir, "e1_recall_per_jarak")
    if df is None or df["resolusi"].nunique() < 2:
        return []
    detectors = list(dict.fromkeys(df["detektor"]))
    resolutions = list(dict.fromkeys(df["resolusi"]))
    fig, axes = plt.subplots(1, len(detectors), figsize=(3.3 * len(detectors), 3.5), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], detectors):
        color, marker = _style(name)
        for resolution, dash in zip(resolutions, ("-", "--", ":")):
            part = df[(df["detektor"] == name) & (df["resolusi"] == resolution)].sort_values("jarak_cm")
            ax.fill_between(part["jarak_cm"], part["recall_low"], part["recall_high"], color=color,
                            alpha=0.10 if dash == "-" else 0.05, linewidth=0)
            ax.plot(part["jarak_cm"], part["recall"], color=color, linestyle=dash, marker=marker,
                    markerfacecolor=color if dash == "-" else SURFACE, label=resolution.replace("x", "×"))
        ax.set_title(_label(name), loc="left")
        ax.set_xlabel("Jarak (cm)")
        ax.set_xticks(sorted(df["jarak_cm"].unique()))
        ax.tick_params(axis="x", labelrotation=45)
    _recall_axis(axes[0][0])
    axes[0][0].legend(loc="lower left")
    fig.suptitle("Pengaruh resolusi terhadap recall (H2)", x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, "grafik4_resolusi")


def plot_pr_curves(out_dir: Path, experiment: str) -> list[Path]:
    """Grafik 5: kurva precision–recall dan AP per detektor."""
    df = _read(out_dir, f"{experiment}_kurva_pr")
    if df is None:
        return []
    resolution = df["resolusi"].iloc[0]
    df = df[df["resolusi"] == resolution]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for name, part in df.groupby("detektor", sort=False):
        color, marker = _style(name)
        ap = f"{part['ap'].iloc[0]:.3f}".replace(".", ",")
        ax.plot(part["recall"], part["precision"], color=color, label=f"{_label(name)} (AP {ap})",
                drawstyle="steps-post")
        ax.plot(part["recall"].iloc[-1], part["precision"].iloc[-1], marker=marker, color=color)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(_comma(1))
    ax.yaxis.set_major_formatter(_comma(1))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    scope = "set jarak" if experiment == "e1" else "set multi-wajah"
    suffix = f", {resolution.replace('x', '×')}" if resolution != "asli" else ""
    ax.set_title(f"Kurva precision–recall ({scope}{suffix})", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=1)
    return _save(fig, out_dir, f"grafik5_kurva_pr_{experiment}")


# ---------------------------------------------------------------------------
# E2, E3, E4
# ---------------------------------------------------------------------------
def plot_count_accuracy(out_dir: Path) -> list[Path]:
    """Grafik 6: akurasi hitung per formasi, batang dengan galat Wilson."""
    df = _read(out_dir, "e2_per_formasi")
    if df is None:
        return []
    formations = list(dict.fromkeys(df["formasi"]))
    detectors = list(dict.fromkeys(df["detektor"]))
    width = 0.8 / len(detectors)
    x = np.arange(len(formations))
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    for i, name in enumerate(detectors):
        color, _ = _style(name)
        part = df[df["detektor"] == name].set_index("formasi").reindex(formations)
        value = part["akurasi_hitung"].to_numpy()
        err = np.vstack([value - part["akurasi_hitung_low"].to_numpy(), part["akurasi_hitung_high"].to_numpy() - value])
        ax.bar(x + (i - (len(detectors) - 1) / 2) * width, value, width * 0.9, color=color, label=_label(name),
               edgecolor=SURFACE, linewidth=1)
        ax.errorbar(x + (i - (len(detectors) - 1) / 2) * width, value, yerr=err, fmt="none", ecolor=INK_2,
                    elinewidth=0.8, capsize=2)
    labels = [f"{f}\n{p.replace(';', '/')}" for f, p in
              zip(formations, df.drop_duplicates("formasi").set_index("formasi").reindex(formations)["posisi_cm"])]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(_comma(1))
    ax.set_ylabel("Akurasi hitung")
    ax.set_xlabel("Formasi (jarak tiap posisi kiri → kanan, cm)")
    ax.set_title("Akurasi hitung per formasi multi-wajah (galat = Wilson 95%)", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=len(detectors))
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "grafik6_akurasi_hitung")


def plot_lighting_heatmap(out_dir: Path) -> list[Path]:
    """Grafik 7: peta panas F1 — varian detektor × cahaya, satu panel per enhancement."""
    df = _read(out_dir, "e3_f1")
    if df is None:
        return []
    enhancements = list(dict.fromkeys(df["enhancement"]))
    variants = list(dict.fromkeys(df["varian"]))
    lightings = list(dict.fromkeys(df["cahaya"]))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("biru", SEQUENTIAL)
    fig, axes = plt.subplots(1, len(enhancements), figsize=(4.2 * len(enhancements) + 0.8, 0.55 * len(variants) + 1.6),
                             sharey=True, squeeze=False)
    image = None
    for ax, enhancement in zip(axes[0], enhancements):
        grid = (df[df["enhancement"] == enhancement].pivot(index="varian", columns="cahaya", values="f1")
                .reindex(index=variants, columns=lightings))
        image = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (row, col), value in np.ndenumerate(grid.to_numpy(dtype=float)):
            text = "–" if np.isnan(value) else f"{value:.2f}".replace(".", ",")
            ax.text(col, row, text, ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if not np.isnan(value) and value > 0.6 else INK)
        ax.set_xticks(range(len(lightings)), lightings)
        ax.set_yticks(range(len(variants)), [_label(v) for v in variants])
        ax.set_title(f"enhancement: {enhancement}", loc="left")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.85, label="F1", format=_comma(1))
    fig.suptitle("F1 per detektor × cahaya × enhancement", x=0.01, y=1.04, ha="left", color=INK, fontsize=11.5)
    return _save(fig, out_dir, "grafik7_f1_cahaya")


def plot_speed(out_dir: Path) -> list[Path]:
    """Grafik 8: waktu per frame, median dengan galat hingga p95, per resolusi."""
    df = _read(out_dir, "e4_kecepatan")
    if df is None:
        return []
    resolutions = list(dict.fromkeys(df["resolusi"]))
    detectors = list(dict.fromkeys(df["detektor"]))
    width = 0.8 / len(detectors)
    x = np.arange(len(resolutions))
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for i, name in enumerate(detectors):
        color, _ = _style(name)
        part = df[df["detektor"] == name].set_index("resolusi").reindex(resolutions)
        median = part["median_ms"].to_numpy()
        offset = x + (i - (len(detectors) - 1) / 2) * width
        device = part["perangkat"].iloc[0]
        ax.bar(offset, median, width * 0.9, color=color, label=f"{_label(name)} — {device}",
               edgecolor=SURFACE, linewidth=1)
        ax.errorbar(offset, median, yerr=[np.zeros_like(median), part["p95_ms"].to_numpy() - median],
                    fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=2)
        for xi, value, top in zip(offset, median, part["p95_ms"].to_numpy()):
            ax.annotate(f"{value:.1f}".replace(".", ","), (xi, top), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, [r.replace("x", "×") for r in resolutions])
    ax.set_ylabel("Waktu per frame (ms)")
    ax.yaxis.set_major_formatter(_comma(0))
    ax.set_title("Waktu deteksi: median, galat hingga persentil ke-95", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "grafik8_kecepatan")


PLOTTERS = {
    "e1": [plot_recall_vs_distance, plot_width_loglog, plot_recall_vs_size, plot_resolution_effect,
           lambda d: plot_pr_curves(d, "e1")],
    "e2": [lambda d: plot_pr_curves(d, "e2"), plot_count_accuracy],
    "e3": [plot_lighting_heatmap],
    "e4": [plot_speed],
    "e5": [],
}


def plot_experiment(name: str, out_dir: Path) -> list[Path]:
    """Bangun semua grafik satu eksperimen dari CSV-nya. Kembalikan path PNG + PDF."""
    _setup()
    written: list[Path] = []
    for plotter in PLOTTERS.get(name, []):
        written += plotter(out_dir)
    return written
