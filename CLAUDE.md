# CLAUDE.md

Konteks permanen untuk Claude Code. Spesifikasi lengkap ada di `docs/PRD.md` (v4) — baca bagian yang relevan sebelum mengerjakan fase apa pun.

## Proyek

Sistem deteksi wajah, crop wajah, dan pengenalan identitas ringan untuk UTS Pengolahan Citra Digital, sekaligus bahan jurnal (PRD **v4**). Membandingkan detektor:

- **OpenCV Haar Cascade** — metode klasik (Viola-Jones)
- **MediaPipe BlazeFace** — model *short-range* dan *full-range*
- **YOLOv8-face lewat OpenCV DNN** — `yolo_n` (P0, YOLOv8n-face derronqi), `yolo_m` (P1, yolov8m-face akanametov seperti repo referensi MariyaSha/FaceRecognition)

lalu memotong setiap wajah per foto (`crop-faces`) dan mengenali **siapa** orangnya dengan **LBPH** (`cv2.face`) — hanya dengan kode pseudonim `S01`, `S02`, … dan hanya untuk peserta yang mencentang izin pengenalan.

Pertanyaan utama: jarak optimal dan ukuran wajah minimum per detektor (RQ1), kinerja pada 2–4 wajah sekaligus (RQ2), pencahayaan dan CLAHE (RQ3), pengaruh resolusi (RQ4), kecepatan (RQ5), titik lemah pose kepala, ekspresi, dan wajah tertutup (RQ6), blur gerak (RQ7), serta akurasi pengenalan LBPH per kondisi (RQ8).

**Pengenalan identitas hanya LBPH berizin** (keputusan v4, 3 Oktober 2026). Tetap tidak ada DeepFace, TensorFlow, *embedding* jaringan saraf, database wajah di luar `data/recognition/`, atau analisis atribut wajah. Versi v2 yang memakai DeepFace tetap batal. Kode v1 tidak dipakai — paket dibangun dari awal.

## Lingkungan

- Selalu kerja di `.venv` (`python3 -m venv .venv`, lalu `source .venv/bin/activate`). Python 3.14 bawaan Mac boleh dipakai.
- **Hanya `opencv-contrib-python>=4.8,<5`.** Jangan pernah memasang `opencv-python` di environment yang sama — keduanya memasang modul `cv2` dan saling menimpa. MediaPipe menarik `opencv-contrib-python` tanpa batas versi; tanpa pin, yang terpasang OpenCV 5 yang tidak menyertakan berkas `haarcascade_*.xml`.
- `mediapipe==1.0.1`, dipatok persis. Di macOS, delegate CPU-nya crash (`Check failed: service_ Service is unavailable` di `TensorsToDetectionsCalculator`), jadi di Mac MediaPipe **selalu** memakai `delegate=GPU` (Metal) dengan citra `SRGBA`. Config `delegate: auto` memilih GPU di macOS dan CPU di Windows/Linux (laptop anggota lain; jalur CPU belum diuji); `PCDFACE_MP_DELEGATE=cpu|gpu` menimpanya. Hasil penelitian hanya dari Mac proyek. Konsekuensinya untuk E4 ada di PRD §8.5.
- Python dari python.org perlu `Install Certificates.command` sekali, kalau tidak unduhan model gagal dengan `CERTIFICATE_VERIFY_FAILED`.
- Model `.tflite` dan `yolov8n-face.onnx` ada di `models/`, diunduh dengan `python -m pcdface download-models` (YOLO dipatok ke satu commit + SHA-256).
- **YOLO hanya lewat OpenCV DNN** (`cv2.dnn.readNetFromONNX`). Paket `ultralytics` menarik `opencv-python` (diuji: versi 5.0) dan PyTorch, jadi **tidak pernah** dipasang di `.venv`. Bobot `.pt` diekspor ke ONNX oleh `python -m pcdface export-yolo`, yang memasang ultralytics di venv terpisah `.venv-yolo-export/`.

## Perintah

```bash
python -m pcdface selftest          # metrik + jalur detektor sintetis, tanpa webcam dan tanpa model
pytest                              # semua tes kecuali yang bertanda models
pytest -m models                    # tes asap yang memanggil MediaPipe dengan berkas .tflite
python -m pcdface validate          # konsistensi data, metadata, anotasi, izin pengenalan
python -m pcdface export-yolo       # yolov8m-face.pt → .onnx di venv terpisah (opsional, untuk yolo_m)
python -m pcdface enroll            # latih LBPH dari foto kondisi acuan peserta berizin → data/recognition/
python -m pcdface crop-faces        # crop setiap wajah per foto (yolo_n); --recognize = kelompokkan per S01/S02/unknown
python -m pcdface run all           # seluruh eksperimen (E1–E8)
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
  detection/               base (DetectionResult), haar, mediapipe_detector, yolo (OpenCV DNN), ycbcr, fake, registry
  recognition/             lbph (LBPH cv2.face, galeri, izin, simpan/muat templat)
  evaluation/              matching, operating_point, average_precision, multiface, distance_analysis, stats
  pose.py                  nama pose (depan, kiri30, menunduk30, …) → sumbu + sudut
  experiments/             e1..e8 + runner (e6 pose/ekspresi/oklusi, e7 blur gerak simulasi, e8 pengenalan LBPH)
  reporting/               tables (CSV+MD), plots (15 grafik §10), report (RINGKASAN.md, contoh gambar terburam)
  tools/                   download_models, export_yolo, capture, annotate, crop, crop_faces, enroll, validate,
                           demo_realtime, forget
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
3. **Memasang `opencv-python`**, `opencv-contrib-python` versi 5, `ultralytics`, atau PyTorch di `.venv`. Ultralytics hanya di `.venv-yolo-export/` lewat `export-yolo`.
4. **Meng-*commit* data wajah** — tidak ada berkas dari `data/` (kecuali `README.md`, `.gitkeep`), termasuk model LBPH di `data/recognition/` (templat biometrik); tidak ada citra dari `results/`; tidak ada `.tflite`, `.onnx`, atau `.pt`.
5. **Menulis nama asli subjek** di mana pun. Hanya ID `S01`, `S02`, ….
6. **Mengubah keputusan PRD §12.2 setelah melihat hasil eksperimen.** Keputusan itu ditetapkan sebelum data diambil.
7. **Menurunkan metrik titik operasi dari run ambang rendah.** Titik operasi dan kurva PR berasal dari dua run terpisah (PRD §8.2).
8. **Memberi skor ≤ 0 ke `cv2.dnn.NMSBoxes`.** Skor ≤ ambang dibuang diam-diam. Geser `levelWeights` Haar menjadi `skor − min + 1` (lihat `detection/haar.py::nms_with_shifted_scores`).
9. **Pengenalan identitas selain LBPH berizin**: DeepFace, TensorFlow, *embedding* jaringan saraf, atau analisis atribut wajah (usia, gender, emosi, ras). LBPH hanya boleh melatih atau menguji wajah peserta dengan `consent_research` **dan** `consent_recognition` = ya, labelnya hanya kode `S01`… (nama ditolak), dan wajah lain selalu `unknown`. Set ekspresi dan oklusi (E6) hanyalah kondisi perekaman yang diperagakan peserta — sistem tidak pernah menebak ekspresi, emosi, atau benda yang dipakai.
10. **Membiarkan model LBPH tetap memuat peserta yang menarik diri atau mencabut izin.** `forget` menghapus model bila memuat peserta itu; `validate` memberi GALAT bila model memuat peserta tanpa izin. Latih ulang dengan `enroll`.



## Cara kerja

- Baca bagian PRD fase yang dikerjakan, lalu langsung kerjakan. Keputusan yang tidak mendesak diambil sendiri dengan default yang masuk akal dan dicatat (PRD §12.2 atau laporan fase); bertanya hanya bila benar-benar memblokir atau berisiko (sudo, ganti versi paket yang dipatok, hapus data nyata).
- Tes yang butuh berkas model diberi `@pytest.mark.models`; tes lain memakai `FakeDetector` dan citra sintetis (peserta sintetis punya pola tahi lalat khas supaya LBPH bisa membedakannya).
- Selesaikan fase dengan `pytest` dan perintah verifikasi fase itu (PRD §14), lalu laporkan hasilnya apa adanya, termasuk yang gagal.
- *Commit* di akhir setiap fase dengan pesan yang menyebut nomor fase.
