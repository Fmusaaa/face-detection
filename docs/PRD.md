# PRD — Sistem Deteksi Wajah: OpenCV vs MediaPipe

**Proyek:** UTS Pengolahan Citra Digital (+ bahan jurnal)
**Versi:** 3.1 — 24 September 2026 (temuan Fase 0: MediaPipe via delegate Metal + `SRGBA`)
**Status:** Siap diimplementasikan lewat Claude Code
**Dokumen terkait:** `CLAUDE.md`, `docs/PROMPT_CLAUDE_CODE.md`, `docs/FORMULIR_PERSETUJUAN.md`

---

## 0. Riwayat Arah Proyek

| Versi | Arah | Status |
|---|---|---|
| v1 | Deteksi klasik: segmentasi warna kulit YCbCr + Haar Cascade | Kode **tidak dipakai** — v3 dibangun dari awal |
| v2 | Deteksi + pengenalan identitas memakai DeepFace | **Dibatalkan** |
| **v3** | **Deteksi saja — OpenCV (Haar Cascade) vs MediaPipe (BlazeFace)** | Dokumen ini |

**Ruang lingkup v3:** sistem hanya menentukan **ada atau tidaknya wajah dan di mana letaknya** (kotak pembatas). Tidak ada pengenalan identitas, tidak ada database wajah, tidak ada DeepFace, tidak ada TensorFlow.

v3 **dibangun dari awal** (keputusan 24 September 2026): tidak ada kode v1 yang disalin dan tidak ada uji regresi v1. Gagasan v1 — enhancement pada kanal Y, detektor warna kulit YCbCr, greedy matching — ditulis ulang di paket `pcdface`.

---

## 1. Arahan yang Harus Dipenuhi

| Kode | Arahan | Dijawab oleh |
|---|---|---|
| **R1** | Pada jarak berapa deteksi bekerja baik | E1, analisis §8.4 |
| **R2** | Wajah di-crop secara manual | Kotak ground truth digambar tangan + ekspor crop (§7) |
| **R3** | Ada pengukuran untuk bahan jurnal | Metrik §8, keluaran §10 |
| **R4** | Data wajah dari anggota kelompok, sebanyak-banyaknya | Protokol dataset §6 |
| **R5** | Perbandingan deteksi untuk beberapa wajah sekaligus | E2 multi-wajah |

---

## 2. Metode yang Dibandingkan

| Kode | Detektor | Jenis | Prioritas |
|---|---|---|---|
| `haar` | Haar Cascade `frontalface_default` (Viola & Jones, 2001) via OpenCV | Klasik — fitur Haar + AdaBoost + cascade | **P0** |
| `mp_short` | MediaPipe Face Detector, model BlazeFace *short-range* | CNN ringan | **P0** |
| `mp_full` | MediaPipe Face Detector, model BlazeFace *full-range* | CNN ringan | **P0** — bila model berhasil diunduh di Fase 0 |
| `mp_sparse` | MediaPipe BlazeFace Sparse *full-range* | CNN, ±60% lebih kecil dari full-range | P1 |
| `ycbcr` | Segmentasi warna kulit YCbCr (Chai & Ngan, 1999) | Klasik — pengolahan citra murni | P1 |

### 2.1 Perbedaan mekanisme — inti pembahasan jurnal

Kedua keluarga detektor punya **cara berbeda memperlakukan ukuran wajah**, dan perbedaan inilah yang membuat perbandingannya menarik:

| | Haar Cascade | MediaPipe BlazeFace |
|---|---|---|
| Cara memindai | Piramida citra pada **resolusi asli**, jendela geser | Seluruh frame **diperkecil** ke ukuran masukan model (128×128 untuk *short-range*) |
| Batas wajah terkecil | **Absolut dalam piksel** — parameter `minSize` (bawaan 30×30) | **Relatif terhadap lebar frame** — wajah selebar 43 px di frame 1280 px tinggal ±4 px di masukan model |
| Jangkauan menurut dokumentasi | Tidak dinyatakan | *Short-range*: "works best for faces within 2 meters"; *full-range*: "best for faces within 5 meters" (dokumentasi MediaPipe Face Detection) |
| Skor keyakinan | Tidak ada secara bawaan — diambil dari `levelWeights` via `detectMultiScale3` | Ada, 0–1 |
| Bentuk kotak | Selalu persegi | Mengikuti wajah |
| Titik landmark | Tidak ada | 6 titik |

### 2.2 Hipotesis

- **H1 — Jarak.** `mp_short` turun tajam di atas ±200 cm sesuai klaim dokumentasinya; `mp_full` bertahan lebih jauh; `haar` turun ketika wajah mendekati 30 px.
- **H2 — Resolusi.** Menaikkan resolusi dari 640×360 ke 1280×720 **membantu Haar** di jarak jauh (wajah punya lebih banyak piksel pada resolusi asli), tetapi **hampir tidak membantu MediaPipe** (frame tetap diperkecil ke ukuran masukan model yang sama).
- **H3 — Pencahayaan.** MediaPipe lebih tahan cahaya redup dan *backlight* dibanding Haar; CLAHE memperkecil selisihnya.
- **H4 — Kecepatan.** MediaPipe lebih cepat per frame dibanding Haar pada resolusi yang sama.

Hipotesis ditulis **sebelum** data diambil. Hasil yang membantah hipotesis tetap dilaporkan apa adanya — itu temuan, bukan kegagalan.

---

## 3. Pertanyaan Penelitian

**RQ1 — Jarak (R1).** Pada rentang jarak berapa setiap detektor bekerja optimal, berapa ukuran wajah minimum (dalam piksel **dan** dalam proporsi lebar frame) agar recall tetap ≥ 0,90, dan apakah klaim jangkauan dokumentasi MediaPipe terbukti?

**RQ2 — Multi-wajah (R5).** Seberapa tepat setiap detektor menghitung dan melokalisasi 2–4 wajah dalam satu frame, termasuk ketika jaraknya berbeda-beda?

**RQ3 — Pencahayaan.** Bagaimana kinerja detektor pada cahaya terang, redup, dan *backlight*, dan apakah CLAHE membantu?

**RQ4 — Resolusi.** Apakah resolusi kamera memengaruhi kedua keluarga detektor secara berbeda (H2)?

**RQ5 — Kecepatan.** Berapa waktu per frame dan FPS masing-masing detektor pada MacBook Air M4?

### Mengapa ukuran wajah dilaporkan dalam piksel dan proporsi

Menurut model kamera lubang jarum, `w_px ≈ f_px × W_wajah / Z` — menggandakan jarak membagi dua lebar wajah di citra. Jarak dalam sentimeter hanya berlaku untuk kamera yang dipakai; ukuran wajah minimum bisa dipakai kamera lain. Karena mekanisme kedua detektor berbeda (§2.1), batas Haar paling tepat dinyatakan dalam **piksel**, sedangkan batas MediaPipe dalam **proporsi lebar frame**. Laporan menyajikan ketiganya.

### Usulan judul

1. *Perbandingan Kinerja Haar Cascade dan MediaPipe BlazeFace untuk Deteksi Wajah Berdasarkan Jarak, Pencahayaan, dan Jumlah Wajah*
2. *Analisis Jarak Operasional dan Ukuran Wajah Minimum pada Deteksi Wajah Metode Klasik dan Berbasis CNN Ringan*
3. *Pengaruh Jarak dan Resolusi Kamera terhadap Akurasi Deteksi Wajah: Studi Komparatif Haar Cascade dan MediaPipe*

---

## 4. Di Luar Ruang Lingkup

| Tidak dikerjakan | Keterangan |
|---|---|
| Pengenalan / identifikasi siapa orangnya | Batasan proyek |
| DeepFace, TensorFlow, *embedding*, database wajah | Keputusan v3 |
| Analisis usia, gender, emosi, ras | — |
| Tracking antar frame | Demo tetap deteksi ulang tiap frame |
| Melatih atau *fine-tune* model | Semua model memakai bobot resmi |
| Penyimpanan cloud | Semua data lokal |

### Batasan masalah (siap tempel ke Bab I)

1. Sistem hanya melakukan deteksi wajah, yaitu menentukan ada atau tidaknya wajah beserta lokasinya dalam citra, tanpa mengenali identitas individu.
2. Metode yang dibandingkan adalah Haar Cascade dari pustaka OpenCV dan BlazeFace dari MediaPipe dengan bobot model resmi tanpa pelatihan ulang.
3. Citra diambil menggunakan webcam laptop pada resolusi 1280×720 piksel, pada jarak 50–300 cm, dalam empat kondisi pencahayaan.
4. Objek uji berupa wajah tampak depan dari anggota kelompok yang telah memberikan persetujuan tertulis, dengan jumlah 1–4 wajah per citra.
5. Sistem tidak melakukan tracking antar frame dan tidak menganalisis atribut wajah.

---

## 5. Arsitektur

```
citra (BGR, OpenCV)
   │
   ├─► enhancement pada kanal Y (none | clahe)
   │
   ├─► DETEKTOR ───────────────────────────────────────────────────────┐
   │     haar       → abu-abu (equalize: on|off) → detectMultiScale(3)  │
   │     mp_short   → BGR→RGBA → mp.Image(SRGBA) → FaceDetector (Metal) │
   │     mp_full    → sama, model full-range                            │
   │     ycbcr (P1) → segmentasi kulit + morfologi + CCL                │
   │                                                                    ▼
   │                                        DetectionResult(boxes, scores, elapsed_ms)
   │                                                                    │
   └─► kotak manual (ground truth) ─────────────► EVALUASI ◄────────────┘
                                                  P/R/F1, IoU, AP, FPPI,
                                                  akurasi hitung, per jarak
```

Semua detektor mengembalikan tipe `DetectionResult` yang sama: `boxes`, `scores` (daftar skor per kotak, `None` bila detektor tidak punya skor), `elapsed_ms`, `stages` (citra antar-tahap, opsional), dan `info` (keterangan khusus detektor, mis. kandidat YCbCr yang ditolak). Dataclass-nya `kw_only`, jadi urutan medan tidak berpengaruh.

### 5.1 Ketentuan implementasi Haar

- Pakai `detectMultiScale3(..., outputRejectLevels=True)` supaya setiap kotak punya `levelWeights` sebagai skor. Skor ini **bukan probabilitas** — tidak berada di rentang 0–1 dan tidak bisa dibandingkan langsung dengan skor MediaPipe. Untuk AP (§8.2) itu tidak masalah karena AP hanya memakai **urutan** skor.
- Untuk run ambang rendah (kurva PR, §8.2): `minNeighbors=0` lalu *non-maximum suppression* dengan `cv2.dnn.NMSBoxes` pada IoU 0,3 — sama dengan `min_suppression_threshold` bawaan MediaPipe, supaya kedua detektor diperlakukan setara. **Jebakan:** `levelWeights` bisa bernilai negatif, sedangkan `NMSBoxes` hanya menyimpan skor yang **lebih besar dari** ambang skor dan **diam-diam membuang** skor negatif (diuji pada OpenCV 4.14 — tanpa pesan galat). Menggeser dengan mengurangkan nilai minimum saja tidak cukup: kandidat terendah menjadi tepat 0 dan ikut terbuang, sehingga citra dengan satu kandidat kehilangan deteksinya. Geser menjadi `skor − min + 1` sebelum NMS; urutan skor tidak berubah dan skor yang dilaporkan tetap `levelWeights` asli. Pada uji awal, 191 kandidat mentah tersisa 7 setelah NMS, dengan tiga wajah asli menempati tiga skor teratas.
- Parameter `equalize` dijadikan faktor eksperimen. Pada uji awal dengan latar abu-abu polos yang lebar, `equalizeHist` global membuat Haar gagal menemukan **satu pun** dari tiga wajah; tanpa ekualisasi ketiganya terdeteksi. Protokol dataset menyarankan latar dinding polos (§6.3), jadi efek ini harus diukur, bukan diasumsikan.
- Kotak Haar selalu persegi, sedangkan kotak manual (garis rambut sampai dagu) lebih tinggi daripada lebar. Walaupun posisinya sempurna, IoU kotak persegi terhadap kotak berasio lebar/tinggi 0,70–0,80 paling tinggi hanya **0,72–0,81**. Ini bukan kesalahan implementasi — sebutkan di Pembahasan saat membandingkan IoU.

### 5.2 Ketentuan implementasi MediaPipe

- **Hanya pakai Tasks API** (`mediapipe.tasks.python.vision.FaceDetector`). Pada MediaPipe 1.0.1, API lama `mp.solutions.face_detection` **sudah tidak ada** — sebagian besar tutorial di internet masih memakainya dan akan gagal dengan `AttributeError`.
- OpenCV membaca citra dalam urutan **BGR**, MediaPipe mengharapkan **RGB**. Konversi dengan `cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)` sebelum membuat `mp.Image(image_format=mp.ImageFormat.SRGBA, data=...)`. Lupa langkah ini adalah galat paling umum dan tidak memunculkan pesan kesalahan — deteksi hanya diam-diam memburuk. Pada uji Fase 0 (`lena.jpg`), masukan BGRA menurunkan skor *short-range* dari 0,845 ke 0,518 dan menyusutkan kotak dari 165 ke 113 px.
- **Delegate GPU wajib di macOS.** Pada MediaPipe 1.0.1 (macOS 27, Apple M4), delegate CPU bawaan langsung *crash* saat detektor dibuat (`graph_service.h:139 Check failed: service_ Service is unavailable`, dari `TensorsToDetectionsCalculator` yang menginisialisasi Metal). Pakai `BaseOptions(..., delegate=BaseOptions.Delegate.GPU)`. Delegate Metal hanya menerima citra 4 kanal — `SRGB` ditolak dengan `unsupported ImageFrame format: 1` — karena itu `SRGBA` di atas.
- `bounding_box` dikembalikan dalam piksel (`origin_x`, `origin_y`, `width`, `height`); potong ke batas citra karena kotak bisa sedikit keluar bingkai.
- `running_mode=IMAGE` untuk eksperimen, `VIDEO` untuk demo realtime. Panggil `close()` setelah selesai.
- Parameter bawaan: `min_detection_confidence=0.5`, `min_suppression_threshold=0.3`.
- Berkas model `.tflite` diunduh sekali ke folder `models/` beserta *checksum* SHA-256-nya:

| Model | URL |
|---|---|
| short-range | `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite` |
| full-range | `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite` |
| sparse (P1) | `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range_sparse.tflite` |

URL full-range dan sparse di atas diambil dari halaman dokumentasi resmi, tetapi belum bisa diverifikasi dengan unduhan langsung saat PRD ini ditulis. Fase 0 wajib memastikan ketiganya benar-benar terunduh; bila full-range gagal, `mp_full` turun ke P1 dan hal itu dicatat.

---

## 6. Protokol Dataset (R4, R5)

### 6.1 Subjek

Semua anggota kelompok yang bersedia, ditambah teman lain yang mau ikut — makin banyak subjek makin sempit interval kepercayaan dan makin beragam wajah yang diuji. Target minimal 5 orang.

Karena tidak ada pengenalan, identitas tidak dipakai oleh sistem. ID pseudonim (`S01`, `S02`, …) tetap diberikan **hanya** untuk mengelompokkan citra saat menghitung interval bootstrap (§8.6) — citra dari orang yang sama saling berkorelasi. Nama asli tidak pernah muncul di nama berkas, metadata, atau hasil.

Setiap subjek menandatangani formulir persetujuan sebelum difoto (§11).

### 6.2 Satu pertemuan cukup

Berbeda dari v2, deteksi tidak membutuhkan sesi pendaftaran dan sesi pengujian terpisah. Seluruh data bisa diambil dalam **satu pertemuan kelompok (±2 jam)**. Set multi-wajah memang butuh semua orang hadir bersamaan, jadi jadwalkan satu pertemuan untuk semuanya.

### 6.3 Pengaturan kamera

| Parameter | Nilai | Alasan |
|---|---|---|
| Resolusi | **1280×720**, tetap | Versi 640×360 dibuat dengan memperkecil citra yang sama (E1), jadi tidak perlu merekam dua kali |
| Posisi kamera | Tetap di meja atau tripod, lensa setinggi mata | Jarak harus konsisten |
| **Center Stage** | **Matikan** | Memotong dan memperbesar citra secara digital untuk mengikuti wajah — ukuran wajah tidak lagi mengecil sesuai jarak, eksperimen jarak jadi tidak sah |
| **Studio Light, Portrait** | **Matikan** | Mengubah pencahayaan dan latar secara digital |
| Latar | Dinding polos, tidak berwarna mirip kulit | Konsisten antar subjek |

Di macOS, ketiga efek video diatur lewat **Control Center → Video Effects** saat kamera aktif. Periksa ulang di awal pertemuan.

### 6.4 Set data

| Set | Isi | Per subjek | Tujuan |
|---|---|---|---|
| **Jarak** | Tanda lakban di lantai pada 50, 100, 150, 200, 250, 300 cm; wajah frontal, cahaya normal; 5 frame per jarak | 30 | E1, E4 |
| **Cahaya** | Di 100 cm: terang, redup, *backlight*; 5 frame per kondisi (normal sudah tercakup set jarak) | 15 | E3 |
| **Multi-wajah** | 6 formasi × 5 frame (tabel di bawah) | — | E2 |
| **Kosong** | 20 frame tanpa wajah: ruangan kosong, meja, benda berwarna mirip kulit (kardus, kayu) | — | Mengukur deteksi palsu |
| **Pose** (P1) | Di 100 cm: menoleh ±30°, menunduk, mendongak; 3 frame per pose | 12 | Pembahasan tambahan |

Formasi multi-wajah — urutan posisi dicatat dari **kiri ke kanan di citra**:

| Formasi | Jumlah wajah | Jarak (cm) |
|---|---|---|
| F1 | 2 | 100, 100 |
| F2 | 3 | 100, 100, 100 |
| F3 | 4 | 150, 150, 150, 150 |
| F4 | 2 | 100, 200 |
| F5 | 3 | 80, 150, 250 |
| F6 | 4 | 80, 130, 200, 300 |

Rotasi siapa berdiri di posisi mana antar frame, supaya satu orang tidak selalu berada di jarak yang sama.

### 6.5 Perkiraan ukuran dataset

Untuk 6 subjek: set jarak 180 citra, set cahaya 90, multi-wajah 30, kosong 20 — total **320 citra berisi 360 wajah**. Anotasi kotak manual kira-kira 10 detik per wajah, jadi sekitar satu jam dan bisa dicicil.

### 6.6 Penamaan berkas dan metadata

```
data/raw/jarak/S03/jarak_S03_150cm_normal_04.jpg
data/raw/cahaya/S03/cahaya_S03_100cm_redup_02.jpg
data/raw/multi/F5/multi_F5_03.jpg
data/raw/kosong/kosong_07.jpg
```

Tool perekam menulis satu baris ke `data/metadata.csv` untuk setiap frame:

| Kolom | Contoh | Keterangan |
|---|---|---|
| `file` | `jarak/S03/jarak_S03_150cm_normal_04.jpg` | Relatif terhadap `data/raw/` |
| `set` | `jarak` | `jarak`, `cahaya`, `multi`, `kosong`, `pose` |
| `subject_id` | `S03` | Kosong untuk `multi` dan `kosong` |
| `formation` | `F5` | Hanya untuk `multi` |
| `positions_cm` | `80;150;250` | Jarak tiap posisi kiri→kanan, hanya untuk `multi` |
| `distance_cm` | `150` | Untuk set satu wajah |
| `lighting` | `normal` | `normal`, `terang`, `redup`, `backlight` |
| `expected_faces` | `3` | Dipakai `validate` untuk mencocokkan jumlah kotak anotasi |
| `luma_mean` | `112.4` | Rerata kanal Y, otomatis — bukti kuantitatif kondisi cahaya |
| `width`, `height` | `1280`, `720` | Validasi resolusi |
| `captured_at` | ISO 8601 | — |

Untuk citra multi-wajah, jarak setiap kotak ditentukan dengan mengurutkan kotak anotasi dari kiri ke kanan berdasarkan koordinat x lalu memasangkannya dengan `positions_cm`. Dengan cara ini recall per jarak di E2 bisa dihitung tanpa perlu menandai jarak satu per satu saat anotasi.

---

## 7. Crop Manual sebagai Ground Truth (R2)

Tool `annotate` dipakai untuk menggambar kotak di **setiap wajah pada setiap citra**. Kotak ini menjadi kebenaran acuan (*ground truth*) seluruh evaluasi. Formatnya JSON — daftar `[x, y, w, h]` per citra, dengan kunci path relatif terhadap `data/raw/`.

Perintah `crop` mengekspor isi setiap kotak ke `data/crops/{set}/...` untuk tiga kegunaan: memeriksa konsistensi anotasi secara visual, mengukur lebar wajah per jarak, dan menyediakan contoh gambar untuk laporan (hanya subjek yang mengizinkan publikasi).

**Aturan kotak — tetapkan sekali, tulis di Metodologi, jangan diubah:** batas atas garis tumbuh rambut, batas bawah ujung dagu, kiri-kanan tepi pipi, tanpa telinga dan leher.

**Konsekuensi yang wajib dibahas:** setiap detektor punya konvensi kotaknya sendiri — Haar selalu persegi, BlazeFace punya proporsinya sendiri — dan keduanya berbeda dari aturan kotak manual. Karena itu IoU dilaporkan pada ambang utama **0,5** dan juga **0,3 serta 0,4** sebagai uji sensitivitas. Bila peringkat detektor berubah antar ambang, perbedaan konvensi kotak ikut berperan dan harus disebutkan.

---

## 8. Metrik (R3)

### 8.1 Pada titik operasi (parameter bawaan detektor)

| Metrik | Definisi |
|---|---|
| TP, FP, FN | Pencocokan greedy pada IoU ≥ ambang: semua pasangan diurutkan menurun menurut IoU, diambil selama kedua anggotanya belum terpakai |
| Precision, Recall, F1 | Dari TP/FP/FN |
| Rerata IoU | Rata-rata IoU pasangan yang cocok — ketepatan letak kotak |
| **FPPI** | *False positives per image* — jumlah deteksi palsu dibagi jumlah citra, termasuk set kosong |

Akurasi tidak dipakai karena *true negative* tidak terdefinisi pada tugas deteksi.

### 8.2 Tanpa ambang: Average Precision

Setiap detektor dijalankan dengan ambang skor rendah supaya semua kandidat tercatat beserta skornya, lalu:

- **Kurva precision–recall** dibentuk dengan mengurutkan seluruh deteksi dari skor tertinggi.
- **AP** = luas di bawah kurva, dengan interpolasi *all-point* seperti PASCAL VOC (Everingham dkk., 2010), pada IoU 0,5.

AP adalah metrik standar di benchmark deteksi wajah seperti WIDER FACE (Yang dkk., 2016). Keunggulannya: tidak bergantung pada pilihan ambang skor, sehingga adil untuk membandingkan Haar (skor `levelWeights`, tanpa batas) dengan MediaPipe (skor 0–1). `ycbcr` tidak punya skor, jadi hanya dilaporkan pada titik operasinya.

Titik operasi (§8.1) dan AP diambil dari **dua run berbeda**: run dengan parameter bawaan, dan run dengan ambang skor rendah. Menurunkan ambang skor MediaPipe bisa mengubah hasil *non-maximum suppression*, jadi metrik titik operasi tidak boleh diturunkan dengan menyaring run ambang rendah.

### 8.3 Multi-wajah

| Metrik | Definisi |
|---|---|
| Akurasi hitung | Proporsi citra dengan jumlah deteksi **tepat sama** dengan jumlah wajah sebenarnya |
| MAE hitung | Rerata selisih absolut jumlah deteksi vs jumlah sebenarnya |
| Recall per posisi jarak | Dari pemasangan kotak ke `positions_cm` (§6.6) |

### 8.4 Jarak (R1)

Untuk setiap jarak dan detektor: recall, lebar wajah (rerata ± simpangan baku, dari kotak manual) dalam piksel, dan lebar wajah sebagai **proporsi lebar frame**.

1. **Validasi model kamera** — regresi `log(w_px)` terhadap `log(jarak)`. Kemiringan mendekati −1 membuktikan `w_px ∝ 1/Z`. Kemiringan yang jauh dari −1 adalah tanda Center Stage masih aktif.
2. **Jarak optimal** — rentang jarak dengan recall ≥ 0,90.
3. **Jarak maksimum efektif** — jarak terbesar yang masih memenuhi ambang itu.
4. **Ukuran wajah minimum** — lebar piksel **dan** proporsi frame terkecil yang masih memberi recall ≥ 0,90.
5. **Uji klaim dokumentasi** — bandingkan recall `mp_short` di ≤200 cm vs >200 cm, dan `mp_full` di seluruh rentang, dengan klaim 2 m dan 5 m.

Bedakan **batas algoritma** dari **batas parameter**: `minSize` Haar 30×30 dan `min_area_ratio` YCbCr (0,2% luas citra ≈ 43×43 px pada 1280×720) membuang wajah kecil secara paksa. Semua parameter batas ukuran dicatat di tabel Metodologi.

### 8.5 Kecepatan

Waktu deteksi per frame diukur dengan `time.perf_counter()`: satu *warm-up* dibuang, minimal 100 pengulangan, dilaporkan **median dan persentil ke-95** beserta FPS, pada 640×360 dan 1280×720. Waktu baca berkas tidak dihitung. Catat perangkat keras (MacBook Air M4) dan versi Python, OpenCV, serta MediaPipe.

**Perangkat eksekusi tidak setara.** Haar berjalan di CPU, sedangkan MediaPipe berjalan di GPU lewat delegate Metal karena delegate CPU-nya tidak bisa dipakai di macOS (§5.2). Perbandingan H4 karena itu adalah perbandingan **implementasi yang tersedia di perangkat ini**, bukan algoritma pada perangkat keras yang sama — tulis ini di Metodologi dan di keterbatasan penelitian. Waktu MediaPipe mencakup konversi BGR→RGBA dan pembuatan `mp.Image`, sama seperti waktu Haar mencakup konversi abu-abu dan ekualisasi.

### 8.6 Interval kepercayaan

- Setiap proporsi (recall, precision, akurasi hitung) disertai **interval Wilson 95%**.
- AP dan F1 disertai **interval bootstrap 95%**, 1000 resampel **per subjek** untuk set satu wajah dan **per citra** untuk set multi-wajah.
- Dua detektor hanya disebut "lebih baik" bila intervalnya tidak tumpang tindih. Bila tumpang tindih, tulis bahwa perbedaannya tidak dapat disimpulkan pada ukuran sampel ini.

---

## 9. Rancangan Eksperimen

Semua faktor, level, dan ambang dibaca dari `configs/experiment.yaml` (Lampiran A).

### E1 — Jarak × resolusi (P0) → RQ1, RQ4, H1, H2

| Faktor | Level |
|---|---|
| Detektor | `haar`, `mp_short`, `mp_full` (P1: `mp_sparse`, `ycbcr`) |
| Resolusi | 1280×720 asli, 640×360 hasil perkecilan skala 0,5 (`cv2.INTER_AREA`, ground truth ikut diskalakan) |
| Jarak | 50–300 cm |
| Data | Set jarak |

Resolusi kecil sengaja dibuat 640×360, bukan 640×480 yang lazim: 640×480 berasio 4:3 sedangkan rekaman 1280×720 berasio 16:9, jadi memperkecil ke 640×480 akan memotong atau meregangkan citra. Skala tepat 0,5 menjaga proporsi dan membuat satu-satunya perbedaan adalah jumlah piksel. Catat alasan ini di Metodologi.

### E2 — Multi-wajah (P0) → RQ2

Detektor P0 × enam formasi pada set multi-wajah. Keluaran: precision, recall, F1, AP, akurasi hitung, MAE hitung, recall per posisi jarak.

### E3 — Pencahayaan × enhancement (P0) → RQ3, H3

| Faktor | Level |
|---|---|
| Detektor | `haar` (equalize on / off), `mp_short`, `mp_full` |
| Enhancement | `none`, `clahe` |
| Cahaya | normal, terang, redup, backlight |
| Data | Set cahaya + set jarak di 100 cm untuk kondisi normal |

### E4 — Kecepatan (P0) → RQ5, H4

Protokol §8.5 untuk semua detektor P0 pada kedua resolusi.

### E5 — Sensitivitas parameter (P1)

Haar: `scaleFactor` {1,05; 1,1; 1,2} × `minNeighbors` {3; 5; 7}. MediaPipe: kurva PR sudah mencakup seluruh rentang `min_detection_confidence`. Keluaran: tabel F1 dan waktu per kombinasi — menunjukkan bahwa perbandingan tidak bergantung pada satu setelan yang kebetulan menguntungkan.

### Ringkasan prioritas

| Eksperimen | Prioritas | Data |
|---|---|---|
| E1 | P0 | Set jarak |
| E2 | P0 | Set multi |
| E3 | P0 | Set cahaya + jarak 100 cm |
| E4 | P0 | Sampel apa saja |
| E5 | P1 | Semua set |

Set kosong ikut dihitung di semua eksperimen untuk FPPI.

---

## 10. Keluaran untuk Laporan dan Jurnal

Setiap eksperimen menulis ke `results/{e1..e5}/`: tabel `.csv` + `.md` (setiap proporsi disertai interval), grafik `.png` 300 dpi + `.pdf`, dan `config_snapshot.yaml` berisi salinan konfigurasi beserta versi perangkat lunak.

Grafik minimum:

1. Recall terhadap jarak per detektor dengan pita interval, plus garis vertikal di 200 cm dan catatan klaim 5 m (E1)
2. Lebar wajah piksel terhadap jarak, skala log-log, dengan garis regresi dan persamaannya (E1)
3. Recall terhadap **proporsi lebar wajah** per detektor — memperlihatkan batas relatif MediaPipe vs batas absolut Haar (E1)
4. Recall terhadap jarak, 1280×720 vs 640×360, dipisah per detektor — menguji H2 (E1)
5. Kurva precision–recall dan nilai AP per detektor (E1 + E2)
6. Akurasi hitung per formasi multi-wajah (E2)
7. Peta panas F1: detektor × cahaya × enhancement (E3)
8. Waktu per frame, median dengan *error bar* p95, per resolusi (E4)

Contoh gambar deteksi hanya memakai wajah subjek yang mencentang izin publikasi; wajah lainnya diburamkan.

---

## 11. Privasi dan Etika Data

Walaupun sistem tidak mengenali siapa pun, foto wajah tetap data pribadi. UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi, Pasal 4 ayat (2), menggolongkan **data biometrik** sebagai data pribadi yang bersifat spesifik, dan penjelasannya menyebut gambar wajah sebagai contoh data biometrik.

| Ketentuan | Implementasi |
|---|---|
| Persetujuan tertulis sebelum difoto | `docs/FORMULIR_PERSETUJUAN.md` |
| Izin publikasi wajah terpisah | Kotak centang terpisah di formulir |
| Pseudonim | `S01`, … ; nama asli tidak disimpan di folder proyek |
| Lokal saja | `data/` dan `results/` di-*gitignore* |
| Hak menarik diri | Perintah `forget S03` menghapus foto, crop, metadata, dan anotasi subjek tersebut. Citra multi-wajah yang memuat subjek itu ikut dihapus |
| Retensi | Data dihapus setelah nilai UTS keluar, atau setelah jurnal terbit bila subjek menyetujui publikasi |

---

## 12. Struktur Proyek

```
pcd-face-detection/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/experiment.yaml
├── docs/
│   ├── PRD.md
│   ├── PROMPT_CLAUDE_CODE.md
│   └── FORMULIR_PERSETUJUAN.md
├── models/                           # .tflite + checksums.txt; di-gitignore kecuali checksums.txt
├── data/                             # di-gitignore kecuali README.md dan .gitkeep
│   ├── raw/{jarak,cahaya,multi,kosong,pose}/
│   ├── metadata.csv
│   ├── annotations/boxes.json
│   └── crops/
├── src/pcdface/
│   ├── __init__.py
│   ├── __main__.py                   # python -m pcdface <perintah>
│   ├── cli.py
│   ├── paths.py                      # PROJECT_ROOT dan path bawaan
│   ├── config.py                     # muat + validasi experiment.yaml
│   ├── preprocessing.py              # enhancement kanal Y (none | clahe), luminansi
│   ├── synthetic.py                  # dataset sintetis lengkap (semua set) untuk --synthetic dan tes
│   ├── dataset/
│   │   ├── metadata.py               # skema metadata.csv + subjects.csv
│   │   ├── annotations.py            # baca/tulis boxes.json
│   │   ├── loader.py                 # gabung metadata + anotasi menjadi sampel
│   │   └── resize.py                 # versi 640×360 + penskalaan ground truth
│   ├── detection/
│   │   ├── base.py                   # DetectionResult(boxes, scores, elapsed_ms, stages)
│   │   ├── haar.py                   # detectMultiScale (titik operasi), detectMultiScale3 + NMS (run AP), flag equalize
│   │   ├── mediapipe_detector.py     # Tasks API, short/full/sparse
│   │   ├── ycbcr.py                  # segmentasi kulit + morfologi + CCL + saring geometri
│   │   ├── fake.py                   # detektor tiruan untuk tes tanpa model
│   │   └── registry.py               # nama → detektor, dibaca dari config
│   ├── evaluation/
│   │   ├── matching.py               # IoU, greedy matching, pencocokan urut skor (AP)
│   │   ├── operating_point.py        # P/R/F1, rerata IoU, FPPI
│   │   ├── average_precision.py      # kurva PR, AP all-point
│   │   ├── multiface.py              # akurasi hitung, MAE, pemasangan posisi
│   │   ├── distance_analysis.py      # regresi log-log, jarak optimal, ukuran minimum
│   │   └── stats.py                  # Wilson, bootstrap per kelompok
│   ├── experiments/
│   │   ├── runner.py                 # snapshot config + versi, jalankan detektor, simpan deteksi
│   │   ├── e1_distance_resolution.py
│   │   ├── e2_multiface.py
│   │   ├── e3_lighting.py
│   │   ├── e4_speed.py
│   │   └── e5_sensitivity.py
│   ├── reporting/
│   │   ├── tables.py
│   │   └── plots.py                  # delapan grafik §10
│   └── tools/
│       ├── download_models.py
│       ├── capture.py                # rekam per set/subjek/jarak/formasi + tulis metadata
│       ├── annotate.py               # gambar kotak ground truth
│       ├── crop.py
│       ├── validate.py
│       ├── demo_realtime.py          # demo webcam, ganti detektor saat berjalan
│       └── forget.py
├── tests/
│   ├── test_config.py
│   ├── test_dataset_tools.py         # metadata, validate, forget pada data sintetis
│   ├── test_matching.py
│   ├── test_operating_point.py
│   ├── test_average_precision.py
│   ├── test_multiface.py
│   ├── test_distance_analysis.py
│   ├── test_stats.py
│   ├── test_detectors_contract.py    # semua detektor mematuhi DetectionResult
│   ├── test_pipeline_fake.py         # jalur penuh dengan FakeDetector
│   └── test_models_smoke.py          # @pytest.mark.models, butuh berkas .tflite
└── results/                          # di-gitignore
```

### 12.1 Perintah

| Perintah | Fungsi |
|---|---|
| `download-models` | Unduh `.tflite`, tulis dan verifikasi SHA-256 |
| `capture --set jarak --subject S03 --distance 150 --lighting normal --count 5` | Rekam dan tulis metadata |
| `capture --set multi --formation F5 --count 5` | Posisi diambil dari tabel formasi di config |
| `annotate` / `crop` / `validate` | Anotasi, ekspor crop, periksa konsistensi data |
| `run e1` … `run e5`, `run all` | Eksperimen; `--synthetic` memakai citra sintetis + FakeDetector |
| `report` | Bangun ulang tabel dan grafik dari hasil tersimpan |
| `demo --detector mp_short` | Demo realtime; tombol untuk ganti detektor saat berjalan |
| `forget S03` | Hapus seluruh data satu subjek |
| `selftest` | Uji seluruh metrik dengan nilai acuan + jalur Haar/YCbCr/Fake pada citra sintetis, tanpa webcam dan tanpa model |

### 12.2 Keputusan implementasi (ditetapkan sebelum data diambil)

Keputusan berikut mengisi celah yang tidak ditentukan bagian lain. Semuanya dibaca dari `configs/experiment.yaml` dan dicatat di `config_snapshot.yaml` setiap run.

1. **Enhancement `none` = citra mentah.** Tidak ada Gaussian blur atau operasi lain. `clahe` hanya pada kanal Y (YCrCb), `clip_limit` 2,0, *tile* 8×8.
2. **Haar `equalize`.** E1, E2, dan E4 memakai nilai config (`true`, praktik umum OpenCV). E3 menguji `true` dan `false`. Nilai ini tidak boleh diubah setelah melihat hasil E3.
3. **Dua jalur Haar.** Titik operasi memakai `detectMultiScale` dengan parameter config (tanpa skor). Run AP memakai `detectMultiScale3(outputRejectLevels=True)` dengan `minNeighbors=0`, lalu NMS pada skor yang sudah digeser (§5.1).
4. **Cakupan metrik.** P/R/F1 dan AP dihitung pada citra berwajah. FPPI dilaporkan dua kali: pada set kosong saja, dan pada seluruh citra (berwajah + kosong).
5. **Interval.** Proporsi → Wilson 95%. Karena frame dari subjek yang sama berkorelasi, recall per jarak juga diberi interval bootstrap per subjek. Perbandingan dua detektor memakai **bootstrap berpasangan atas selisih** (ΔF1, ΔAP, Δrecall) pada resampel subjek yang sama; "lebih baik" hanya bila interval selisihnya tidak memuat 0.
6. **Jarak optimal** = jarak dengan estimasi titik recall ≥ `recall_target`; batas bawah Wilson ikut dilaporkan. **Ukuran wajah minimum** = batas bawah bin lebar wajah (dari kotak manual, kedua resolusi digabung) terkecil sehingga bin itu dan semua bin di atasnya punya recall ≥ `recall_target`. Bin piksel dan bin proporsi ada di config.
7. **Metadata** ditambah kolom `subjects` (kode peserta kiri→kanan pada citra multi-wajah; dibutuhkan `forget`) dan `pose`. Izin publikasi disimpan di `data/subjects.csv` (kode + izin, tanpa nama).
8. **Waktu deteksi** mencakup konversi warna yang dibutuhkan detektor (BGR→abu-abu+ekualisasi untuk Haar, BGR→RGBA+`mp.Image` untuk MediaPipe), tidak mencakup baca berkas maupun enhancement.

---

## 13. Lingkungan

### 13.1 Python

MediaPipe 1.0.1 dirilis sebagai *wheel* `py3-none-macosx_11_0_arm64`, yang tidak terikat versi Python tertentu. Kombinasi MediaPipe 1.0.1 + OpenCV contrib 4.14 sudah diuji bisa di-*import* dan dipakai membuat `mp.Image` pada Python 3.14 — jadi **Python 3.14 bawaan Mac boleh dipakai**, tidak perlu memasang Python 3.12.

Tetap wajib memakai *virtual environment*, karena alasan di §13.2:

```bash
cd ~/pcd-face-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import cv2, mediapipe; print(cv2.__version__, mediapipe.__version__)"
```

Versi OpenCV yang tercetak harus diawali **4**.

### 13.2 requirements.txt

```
opencv-contrib-python>=4.8,<5   # JANGAN pasang opencv-python di environment yang sama
mediapipe==1.0.1
numpy>=1.26
pandas>=2.0
matplotlib>=3.8
pyyaml>=6.0
pytest>=8.0
```

**Kenapa `opencv-contrib-python`, bukan `opencv-python`:** MediaPipe 1.0.1 mencantumkan `opencv-contrib-python` sebagai dependensi **tanpa batas versi**. Tanpa pin, pip memasang OpenCV contrib 5.0 — yang masih punya `CascadeClassifier` tetapi **tidak lagi menyertakan berkas `haarcascade_*.xml`**, sehingga detektor Haar gagal dengan `FileNotFoundError`. Selain itu `opencv-python` dan `opencv-contrib-python` sama-sama memasang modul `cv2` dan saling menimpa bila dipasang bersamaan. Solusinya: pasang **hanya** `opencv-contrib-python` dengan batas `<5`. Kombinasi ini sudah diuji: OpenCV 4.14 terpasang dan berkas Haar tersedia.

`opencv-python` 4.14 yang sekarang terpasang di Python global tidak mengganggu, selama semua pekerjaan dilakukan di dalam `.venv`.

---

## 14. Kriteria Selesai per Fase

| Fase | Isi | Verifikasi |
|---|---|---|
| **0 — Lingkungan** | `.venv`, requirements, unduh model | OpenCV 4.x; berkas Haar ada; tiga `.tflite` terunduh dengan SHA-256 tercatat (atau full-range/sparse dicatat gagal); `FaceDetector` short-range berhasil dibuat dan dijalankan pada satu citra |
| **1 — Kerangka paket** | Paket `pcdface` dari awal, `paths`, `config`, `DetectionResult`, preprocessing, matching, CLI kerangka | `pytest` lulus; `python -m pcdface selftest` lulus |
| **2 — Alat data** | capture (semua set), metadata, annotate, crop, validate, forget | Metadata benar untuk tiap set; `validate` menangkap jumlah kotak ≠ `expected_faces` dan berkas yatim; `forget` menghapus tuntas termasuk citra multi-wajah |
| **3 — Detektor & metrik** | haar (skor + equalize), mediapipe, fake, registry, seluruh modul evaluasi | Tes metrik lulus dengan nilai acuan hitungan tangan; tes kontrak detektor lulus; `pytest -m models` lulus |
| **4 — Eksperimen** | E1–E4 (+E5), runner, perkecilan resolusi | `run all --synthetic` menghasilkan semua tabel §10 tanpa galat |
| **5 — Pelaporan & demo** | tables, plots, demo dengan MediaPipe | Delapan grafik terbentuk; demo bisa berganti haar ↔ mp_short ↔ mp_full saat berjalan |

Sepanjang semua fase: `pytest` hijau, dan tidak ada berkas dari `data/`, `models/*.tflite`, atau citra `results/` yang ter-*commit*.

---

## 15. Jadwal Menuju UTS

Sesuaikan dengan tanggal UTS sebenarnya.

| Minggu | Tanggal | Target |
|---|---|---|
| 1 | 23–29 Sep | Fase 0–1; formulir persetujuan ditandatangani |
| 2 | 30 Sep–6 Okt | Fase 2; **satu pertemuan kelompok** untuk seluruh pengambilan data; mulai anotasi |
| 3 | 7–13 Okt | Fase 3; selesaikan anotasi dan crop |
| 4 | 14–20 Okt | Fase 4; jalankan E1–E4 |
| 5 | 21–27 Okt | Fase 5; susun draf paper; latihan demo |

---

## 16. Risiko dan Mitigasi

| Risiko | Mitigasi |
|---|---|
| Mengikuti tutorial lama `mp.solutions.face_detection` | Hanya Tasks API (§5.2); tercantum di aturan CLAUDE.md |
| OpenCV 5 terpasang lewat dependensi MediaPipe | Pin `opencv-contrib-python<5` (§13.2) |
| Lupa konversi BGR→RGB | Dikunci di `mediapipe_detector.py` + tes yang gagal bila urutan kanal salah |
| MediaPipe 1.0.1 *crash* dengan delegate CPU di macOS | Delegate GPU (Metal) + `SRGBA` (§5.2); ketidaksetaraan CPU vs GPU dibahas di §8.5 |
| Unduhan model gagal `CERTIFICATE_VERIFY_FAILED` (Python python.org) | Jalankan `Install Certificates.command` sekali |
| Model `.tflite` gagal diunduh di jaringan kampus | Unduh di Fase 0 lewat jaringan rumah; *checksum* dicatat |
| Center Stage aktif | Checklist §6.3; regresi log-log §8.4 sebagai pendeteksi |
| `equalizeHist` menggagalkan Haar pada latar polos | Jadikan faktor E3, bukan asumsi |
| Konvensi kotak berbeda antar detektor | IoU 0,3 / 0,4 / 0,5 dan AP (§7, §8.2) |
| Jumlah subjek kecil | Interval kepercayaan dilaporkan jujur (§8.6) |
| Wajah tidak terdeteksi di jarak jauh | Itu **temuan** yang menjawab R1, bukan galat |

---

## 17. Referensi

- Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). BlazeFace: Sub-millisecond neural face detection on mobile GPUs. *arXiv:1907.05047*.
- Chai, D., & Ngan, K. N. (1999). Face segmentation using skin-color map in videophone applications. *IEEE Transactions on Circuits and Systems for Video Technology*, 9(4), 551–564.
- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision*, 88(2), 303–338.
- Google AI Edge. *Face detection guide — MediaPipe*. https://developers.google.com/edge/mediapipe/solutions/vision/face_detector
- Lugaresi, C., dkk. (2019). MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*.
- Undang-Undang Republik Indonesia Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi.
- Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *Proceedings of CVPR 2001*.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.
- Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). WIDER FACE: A face detection benchmark. *Proceedings of CVPR 2016*.
- Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. *Graphics Gems IV*, 474–485.

---

## Lampiran A — `configs/experiment.yaml`

Salinan isi config pada saat PRD ini diperbarui. Bila berbeda, **berkas config yang berlaku**.

```yaml
# Konfigurasi eksperimen — satu-satunya sumber faktor, level, ambang, dan seed.
# Salinannya ditulis ke results/<eksperimen>/config_snapshot.yaml setiap run.
# Keputusan yang ditetapkan sebelum data diambil: docs/PRD.md §12.2.

seed: 42

paths:                               # relatif terhadap akar proyek
  raw: data/raw
  metadata: data/metadata.csv
  subjects: data/subjects.csv
  annotations: data/annotations/boxes.json
  crops: data/crops
  models: models
  results: results

capture:
  width: 1280
  height: 720
  camera_index: 0

dataset:
  sets: [jarak, cahaya, multi, kosong, pose]
  distances_cm: [50, 100, 150, 200, 250, 300]
  lightings: [normal, terang, redup, backlight]
  poses: [kiri, kanan, menunduk, mendongak]
  reference_distance_cm: 100         # jarak set cahaya dan pose
  frames_per_condition: 5
  empty_images: 20
  formations:                        # posisi kiri → kanan di citra
    F1: [100, 100]
    F2: [100, 100, 100]
    F3: [150, 150, 150, 150]
    F4: [100, 200]
    F5: [80, 150, 250]
    F6: [80, 130, 200, 300]

preprocessing:
  clahe_clip_limit: 2.0
  clahe_tile_grid: [8, 8]

detectors:
  haar:
    type: haar
    scale_factor: 1.1
    min_neighbors: 5
    min_size: [30, 30]
    equalize: true                   # E3 menguji true dan false (PRD §12.2 butir 2)
  mp_short:
    type: mediapipe
    model: blaze_face_short_range.tflite
    min_detection_confidence: 0.5
    min_suppression_threshold: 0.3
  mp_full:
    type: mediapipe
    model: blaze_face_full_range.tflite
    min_detection_confidence: 0.5
    min_suppression_threshold: 0.3
  mp_sparse:                         # P1
    type: mediapipe
    model: blaze_face_full_range_sparse.tflite
    min_detection_confidence: 0.5
    min_suppression_threshold: 0.3
  ycbcr:                             # P1 — Chai & Ngan (1999)
    type: ycbcr
    cb_range: [77, 127]
    cr_range: [133, 173]
    opening_kernel: 5
    closing_kernel: 11
    opening_iterations: 1
    closing_iterations: 2
    min_area_ratio: 0.002            # ≈ 43×43 px pada 1280×720
    max_area_ratio: 0.5
    aspect_range: [0.55, 1.30]       # lebar / tinggi
    min_solidity: 0.45

evaluation:
  iou_primary: 0.5
  iou_sensitivity: [0.3, 0.4]
  recall_target: 0.90
  ap_run:                            # run ambang rendah untuk kurva PR (PRD §8.2)
    mp_min_detection_confidence: 0.05
    haar_min_neighbors: 0
    haar_nms_iou: 0.3                # samakan dengan min_suppression_threshold MediaPipe
  size_bins_px: [0, 20, 30, 40, 50, 60, 80, 100, 150, 250, 10000]
  size_bins_ratio: [0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.25, 1.0]

experiments:
  e1:
    detectors: [haar, mp_short, mp_full]
    resolutions: [[1280, 720], [640, 360]]
  e2:
    detectors: [haar, mp_short, mp_full]
  e3:
    detectors: [haar, mp_short, mp_full]
    haar_equalize: [true, false]
    enhancements: [none, clahe]
  e4:
    detectors: [haar, mp_short, mp_full]
    resolutions: [[1280, 720], [640, 360]]
    warmup_runs: 1
    repeats: 100
    sample_images: 10                # citra per set, diambil acak dengan seed
  e5:
    enabled: false                   # P1
    scale_factors: [1.05, 1.1, 1.2]
    min_neighbors: [3, 5, 7]

stats:
  ci_level: 0.95
  bootstrap_resamples: 1000

synthetic:                           # hanya untuk --synthetic dan tes
  subjects: 4
  frames_per_condition: 2
  empty_images: 6
  focal_px: 1000.0                   # model lubang jarum: w_px = f × W / Z
  face_width_cm: 15.0
```
