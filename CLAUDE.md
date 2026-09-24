# CLAUDE.md

Konteks permanen untuk Claude Code. Spesifikasi lengkap ada di `docs/PRD.md` (v3) — baca bagian yang relevan sebelum mengerjakan fase apa pun.

## Proyek

Sistem **deteksi wajah saja** untuk UTS Pengolahan Citra Digital, sekaligus bahan jurnal. Membandingkan:

- **OpenCV Haar Cascade** — metode klasik (Viola-Jones)
- **MediaPipe BlazeFace** — model *short-range* dan *full-range*

Pertanyaan utama: jarak optimal dan ukuran wajah minimum per detektor (RQ1), kinerja pada 2–4 wajah sekaligus (RQ2), pencahayaan dan CLAHE (RQ3), pengaruh resolusi (RQ4), kecepatan (RQ5).

**Tidak ada pengenalan identitas.** Tidak ada DeepFace, TensorFlow, *embedding*, atau database wajah. Versi v2 yang memakai DeepFace sudah dibatalkan. Kode v1 tidak dipakai — paket dibangun dari awal.

## Lingkungan

- Selalu kerja di `.venv` (`python3 -m venv .venv`, lalu `source .venv/bin/activate`). Python 3.14 bawaan Mac boleh dipakai.
- **Hanya `opencv-contrib-python>=4.8,<5`.** Jangan pernah memasang `opencv-python` di environment yang sama — keduanya memasang modul `cv2` dan saling menimpa. MediaPipe menarik `opencv-contrib-python` tanpa batas versi; tanpa pin, yang terpasang OpenCV 5 yang tidak menyertakan berkas `haarcascade_*.xml`.
- `mediapipe==1.0.1`, dipatok persis. Di macOS, delegate CPU-nya crash (`Check failed: service_ Service is unavailable` di `TensorsToDetectionsCalculator`), jadi MediaPipe **selalu** memakai `delegate=GPU` (Metal) dengan citra `SRGBA`. Konsekuensinya untuk E4 ada di PRD §8.5.
- Python dari python.org perlu `Install Certificates.command` sekali, kalau tidak unduhan model gagal dengan `CERTIFICATE_VERIFY_FAILED`.
- Model `.tflite` ada di `models/`, diunduh dengan `python -m pcdface download-models`.

## Perintah

```bash
python -m pcdface selftest          # metrik + jalur detektor sintetis, tanpa webcam dan tanpa model
pytest                              # semua tes kecuali yang bertanda models
pytest -m models                    # tes asap yang memanggil MediaPipe dengan berkas .tflite
python -m pcdface validate          # konsistensi data, metadata, anotasi
python -m pcdface run all           # seluruh eksperimen
python -m pcdface run all --synthetic   # uji jalur eksperimen dengan citra sintetis + FakeDetector
python -m pcdface report            # bangun ulang tabel dan grafik
```

## Peta kode

```
src/pcdface/
  paths.py, config.py      path bawaan dari PROJECT_ROOT; configs/experiment.yaml
  preprocessing.py         enhancement pada kanal Y (none | clahe)
  synthetic.py             dataset sintetis untuk --synthetic dan tes
  dataset/                 metadata.csv, versi resolusi 640×360
  detection/               base (DetectionResult), haar, mediapipe_detector, ycbcr, fake, registry
  evaluation/              matching, operating_point, average_precision, multiface, distance_analysis, stats
  experiments/             e1..e5 + runner
  reporting/               tables (CSV+MD), plots (8 grafik §10), report (RINGKASAN.md, contoh gambar terburam)
  tools/                   download_models, capture, annotate, crop, validate, demo_realtime, forget
```

## Konvensi

- Identifier dalam bahasa Inggris. Docstring, komentar, pesan CLI, dan label grafik dalam bahasa Indonesia.
- Type hints di semua fungsi publik; `dataclass` untuk struktur hasil.
- Semua path dari `paths.py`; semua faktor, level, ambang, dan seed dari `configs/experiment.yaml`.
- Semua detektor mengembalikan `DetectionResult(boxes, scores, elapsed_ms, stages)`. `scores=None` bila detektor tidak punya skor.
- Setiap proporsi yang dilaporkan disertai interval Wilson 95%; AP dan F1 disertai bootstrap per subjek (set satu wajah) atau per citra (multi-wajah).

## Aturan keras — jangan pernah

1. **Memakai `mp.solutions`** atau mengikuti tutorial MediaPipe lama. MediaPipe 1.0.1 hanya punya Tasks API (`mediapipe.tasks.python.vision.FaceDetector`).
2. **Mengumpankan citra BGR ke MediaPipe.** Selalu `cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)` sebelum `mp.Image(image_format=mp.ImageFormat.SRGBA, ...)`. `SRGB` tiga kanal ditolak delegate Metal (`unsupported ImageFrame format: 1`).
3. **Memasang `opencv-python`**, atau `opencv-contrib-python` versi 5.
4. **Meng-*commit* data wajah** — tidak ada berkas dari `data/` (kecuali `README.md`, `.gitkeep`), tidak ada citra dari `results/`, tidak ada `.tflite`.
5. **Menulis nama asli subjek** di mana pun. Hanya ID `S01`, `S02`, ….
6. **Mengubah keputusan PRD §12.2 setelah melihat hasil eksperimen.** Keputusan itu ditetapkan sebelum data diambil.
7. **Menurunkan metrik titik operasi dari run ambang rendah.** Titik operasi dan kurva PR berasal dari dua run terpisah (PRD §8.2).
8. **Memberi skor ≤ 0 ke `cv2.dnn.NMSBoxes`.** Skor ≤ ambang dibuang diam-diam. Geser `levelWeights` Haar menjadi `skor − min + 1` (lihat `detection/haar.py::nms_with_shifted_scores`).
9. **Menambahkan pengenalan identitas**, DeepFace, TensorFlow, atau analisis atribut wajah.

## Cara kerja

- Baca bagian PRD fase yang dikerjakan, lalu langsung kerjakan. Keputusan yang tidak mendesak diambil sendiri dengan default yang masuk akal dan dicatat (PRD §12.2 atau laporan fase); bertanya hanya bila benar-benar memblokir atau berisiko (sudo, ganti versi paket yang dipatok, hapus data nyata).
- Tes yang butuh berkas model diberi `@pytest.mark.models`; tes lain memakai `FakeDetector` dan citra sintetis.
- Selesaikan fase dengan `pytest` dan perintah verifikasi fase itu (PRD §14), lalu laporkan hasilnya apa adanya, termasuk yang gagal.
- *Commit* di akhir setiap fase dengan pesan yang menyebut nomor fase.
