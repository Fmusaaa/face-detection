# Deteksi, Crop, dan Pengenalan Wajah: Haar Cascade vs MediaPipe BlazeFace vs YOLO

Proyek UTS Pengolahan Citra Digital (+ bahan jurnal). Membandingkan **OpenCV Haar Cascade**,
**MediaPipe BlazeFace** (short-range, full-range), dan **YOLOv8-face lewat OpenCV DNN** berdasarkan
jarak, jumlah wajah, pencahayaan, resolusi, kecepatan, pose kepala, ekspresi, penutup wajah, dan
blur gerak; memotong setiap wajah per foto; lalu mengenali **siapa** orangnya dengan **LBPH**
(seperti repo [MariyaSha/FaceRecognition](https://github.com/MariyaSha/FaceRecognition)) — hanya
dengan kode peserta `S01`, `S02`, … dan hanya untuk peserta yang mengizinkan.

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
- **Hanya** `opencv-contrib-python<5` — jangan pasang `opencv-python`, `ultralytics`, atau PyTorch di `.venv`
  yang sama. YOLO dijalankan lewat OpenCV DNN (`yolov8n-face.onnx` ikut diunduh `download-models`).
- Opsional — model repo referensi (`yolov8m-face`, detektor `yolo_m`): `python -m pcdface export-yolo`.
  Alat ini membuat venv terpisah `.venv-yolo-export/` berisi ultralytics (±1 GB), mengekspor `.pt` ke
  `.onnx`, lalu memeriksanya dengan OpenCV DNN. `.venv` utama tidak tersentuh.
- MediaPipe 1.0.1 di macOS berjalan lewat GPU (Metal); delegate CPU-nya crash (PRD §5.2).
- **Windows / Linux:** `delegate: auto` otomatis memakai CPU. Aktivasi venv di Windows:
  `.venv\Scripts\activate`. Jalur ini belum pernah diuji — jalankan `pytest` dan
  `python -m pcdface run all --synthetic` dulu, lalu kabari bila gagal. Angka hasil penelitian
  tetap dijalankan di Mac proyek (PRD §12.2 butir 11).


## Hari pengambilan data

1. Setiap peserta menandatangani [formulir persetujuan](docs/FORMULIR_PERSETUJUAN.md) **versi 4**. Formulir
   berisi nama asli — simpan di luar folder proyek. Bagian 7 (izin pengenalan identitas) terpisah;
   peserta yang sudah menandatangani versi lama perlu mengisi bagian 7 sebelum wajahnya dipakai `enroll`/E8.
2. Isi `data/subjects.csv` — kode peserta dan izinnya saja, **tanpa nama** (lihat [`data/README.md`](data/README.md)).
3. Matikan **Center Stage, Studio Light, Portrait** (Control Center → Video Effects saat kamera aktif).
   Tempel lakban di lantai pada 50, 100, 150, 200, 250, 300 cm, dan titik penanda toleh di dinding
   setinggi mata (30° dan 60°, PRD §6.4).
4. Rekam. Data boleh diambil di beberapa pertemuan — beri satu kode sesi per pertemuan dan rekam
   **seluruh set jarak, cahaya, pose, ekspresi, dan oklusi seorang peserta dalam sesi yang sama** (PRD §6.2).
   Siapkan satu masker dan satu kacamata hitam. Contoh untuk satu peserta (±40 menit):

```bash
SESI=2026-10-01-sore
for d in 50 100 150 200 250 300; do python -m pcdface capture --session $SESI --set jarak --subject S01 --distance $d; done
for l in terang redup backlight; do python -m pcdface capture --session $SESI --set cahaya --subject S01 --lighting $l; done
for d in 100 200; do python -m pcdface capture --session $SESI --set pose --subject S01 --distance $d --pose semua; done
python -m pcdface capture --session $SESI --set ekspresi --subject S01 --expression semua
python -m pcdface capture --session $SESI --set oklusi --subject S01 --occlusion semua
```

   `--pose/--expression/--occlusion semua` merekam semua level berurutan (3 frame per level);
   instruksi berikutnya tampil di bagian atas layar. Kiri/kanan = kiri/kanan **peserta**.

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
python -m pcdface enroll            # latih LBPH dari foto kondisi acuan peserta berizin → data/recognition/
python -m pcdface run all           # E1–E8 (E7 blur simulasi, E8 pengenalan LBPH)
python -m pcdface report            # results/RINGKASAN.md + 15 grafik
python -m pcdface report --examples 4   # contoh gambar; wajah tanpa izin publikasi diburamkan
python -m pcdface demo --detector yolo_n   # demo webcam; 1/2/3/4 atau d = ganti detektor, r = kenali
```

### Crop wajah per foto (YOLO + OpenCV)

```bash
python -m pcdface crop-faces                               # semua foto data/raw/ → results/crop_wajah/
python -m pcdface crop-faces --input foto_kelas/ --margin 0.2
python -m pcdface crop-faces --input foto_kelas/ --recognize   # crop dikelompokkan ke S01/, S02/, …, unknown/
```

Setiap wajah menjadi `<nama foto>_f<N>.jpg` (N urut kiri → kanan) dan dicatat di `crop_wajah.csv`
(kotak, skor, identitas, jarak LBPH). Gaya repo referensi juga didukung: `python -m pcdface enroll
--folder faces/` dengan subfolder `faces/S01/`, `faces/S02/` (nama folder harus kode peserta, bukan nama).

Keluaran per eksperimen di `results/<eN>/`: tabel `.csv` + `.md` (interval Wilson / bootstrap),
grafik `.png` 300 dpi + `.pdf`, `config_snapshot.yaml` (config + versi perangkat lunak), `deteksi.jsonl`.

Uji jalur tanpa data asli: `python -m pcdface run all --synthetic && python -m pcdface report --synthetic`
(hasil di `results/synthetic/`, angkanya **bukan** hasil penelitian).

## Privasi

- Nama asli tidak pernah masuk ke folder proyek; hanya kode `S01`, `S02`, ….
- `data/` dan `results/` tidak pernah di-commit (lihat `.gitignore`).
- Peserta menarik diri: `python -m pcdface forget S03` — menghapus foto, foto multi-wajah yang
  memuatnya, crop, metadata, anotasi, barisnya di `subjects.csv`, dan model LBPH bila memuatnya.
  Lalu jalankan ulang `enroll`, `run all`, dan `report`.
- Model LBPH (`data/recognition/`) adalah templat biometrik: tidak pernah di-commit atau dibagikan.
  Hanya peserta dengan `consent_recognition = ya` yang masuk; `validate` memberi GALAT bila model memuat
  peserta tanpa izin. Wajah yang tidak cocok selalu berlabel `unknown` — sistem tidak pernah menebak nama.

## Tes

```bash
pytest               # tanpa model: metrik (nilai acuan hitungan tangan), alat data, jalur sintetis
pytest -m models     # butuh models/*.tflite + yolov8n-face.onnx; PCDFACE_SMOKE_IMAGE=<foto satu wajah>
```

## Struktur

```
configs/experiment.yaml   semua faktor, level, ambang, seed
src/pcdface/              paket (lihat CLAUDE.md § Peta kode)
tests/                    pytest
data/                     foto, metadata, anotasi — lokal saja
models/                   .tflite, .onnx, .pt (lokal) + checksums.txt
results/                  keluaran eksperimen — lokal saja
```
