# Deteksi Wajah: Haar Cascade vs MediaPipe BlazeFace

Proyek UTS Pengolahan Citra Digital (+ bahan jurnal). Membandingkan **OpenCV Haar Cascade** dengan
**MediaPipe BlazeFace** (short-range, full-range) berdasarkan jarak, jumlah wajah, pencahayaan,
resolusi, dan kecepatan. **Hanya deteksi** — tidak ada pengenalan identitas.

Spesifikasi lengkap: [`docs/PRD.md`](docs/PRD.md). Konteks untuk Claude Code: [`CLAUDE.md`](CLAUDE.md).

## Persiapan (sekali)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pcdface download-models
python -m pcdface selftest
```

- Python dari python.org perlu menjalankan `Install Certificates.command` sekali, kalau tidak
  `download-models` gagal dengan `CERTIFICATE_VERIFY_FAILED`.
- **Hanya** `opencv-contrib-python<5` — jangan pasang `opencv-python` di `.venv` yang sama.
- MediaPipe 1.0.1 di macOS berjalan lewat GPU (Metal); delegate CPU-nya crash (PRD §5.2).

## Hari pengambilan data

1. Setiap peserta menandatangani [formulir persetujuan](docs/FORMULIR_PERSETUJUAN.md). Formulir berisi
   nama asli — simpan di luar folder proyek.
2. Isi `data/subjects.csv` — kode peserta dan izinnya saja, **tanpa nama** (lihat [`data/README.md`](data/README.md)).
3. Matikan **Center Stage, Studio Light, Portrait** (Control Center → Video Effects saat kamera aktif).
   Tempel lakban di lantai pada 50, 100, 150, 200, 250, 300 cm, dan titik penanda toleh di dinding
   setinggi mata (30° dan 60°, PRD §6.4).
4. Rekam. Data boleh diambil di beberapa pertemuan — beri satu kode sesi per pertemuan dan rekam
   **seluruh set jarak, cahaya, pose, dan ekspresi seorang peserta dalam sesi yang sama** (PRD §6.2).
   Contoh untuk satu peserta (±35 menit):

```bash
SESI=2026-10-01-sore
for d in 50 100 150 200 250 300; do python -m pcdface capture --session $SESI --set jarak --subject S01 --distance $d; done
for l in terang redup backlight; do python -m pcdface capture --session $SESI --set cahaya --subject S01 --lighting $l; done
for d in 100 200; do python -m pcdface capture --session $SESI --set pose --subject S01 --distance $d --pose semua; done
python -m pcdface capture --session $SESI --set ekspresi --subject S01 --expression semua
```

   `--pose semua` dan `--expression semua` merekam semua level berurutan (3 frame per level);
   instruksi pose/ekspresi berikutnya tampil di bagian atas layar. Kiri/kanan = kiri/kanan **peserta**.

   Multi-wajah — urutan `--subjects` adalah urutan **kiri → kanan di layar** (pratinjau tidak dicerminkan):

```bash
python -m pcdface capture --session $SESI --set multi --formation F5 --subjects S02,S05,S01
python -m pcdface capture --session $SESI --set kosong --count 20
```

   Tombol: SPASI simpan, q keluar. Frame yang bukan 1280×720 ditolak; peserta yang belum terdaftar
   di `subjects.csv` tidak bisa direkam.

## Setelah pengambilan data

```bash
python -m pcdface annotate          # kotak manual: garis rambut → dagu, pipi → pipi, tanpa telinga
python -m pcdface validate          # harus 0 galat sebelum eksperimen
python -m pcdface crop              # ekspor crop + lebar wajah per jarak
python -m pcdface run all           # E1–E4 dan E6 (E5 bila experiments.e5.enabled)
python -m pcdface report            # results/RINGKASAN.md + 10 grafik

python -m pcdface report --examples 4   # contoh gambar; wajah tanpa izin publikasi diburamkan
python -m pcdface demo              # demo webcam; 1/2/3 atau d = ganti detektor
```

Keluaran per eksperimen di `results/<eN>/`: tabel `.csv` + `.md` (interval Wilson / bootstrap),
grafik `.png` 300 dpi + `.pdf`, `config_snapshot.yaml` (config + versi perangkat lunak), `deteksi.jsonl`.

Uji jalur tanpa data asli: `python -m pcdface run all --synthetic && python -m pcdface report --synthetic`
(hasil di `results/synthetic/`, angkanya **bukan** hasil penelitian).

## Privasi

- Nama asli tidak pernah masuk ke folder proyek; hanya kode `S01`, `S02`, ….
- `data/` dan `results/` tidak pernah di-commit (lihat `.gitignore`).
- Peserta menarik diri: `python -m pcdface forget S03` — menghapus foto, foto multi-wajah yang
  memuatnya, crop, metadata, anotasi, dan barisnya di `subjects.csv`. Lalu jalankan ulang `run all` dan `report`.

## Tes

```bash
pytest               # tanpa model: metrik (nilai acuan hitungan tangan), alat data, jalur sintetis
pytest -m models     # butuh models/*.tflite; PCDFACE_SMOKE_IMAGE=<foto satu wajah> untuk uji urutan kanal
```

## Struktur

```
configs/experiment.yaml   semua faktor, level, ambang, seed
src/pcdface/              paket (lihat CLAUDE.md § Peta kode)
tests/                    pytest
data/                     foto, metadata, anotasi — lokal saja
models/                   .tflite (lokal) + checksums.txt
results/                  keluaran eksperimen — lokal saja
```
