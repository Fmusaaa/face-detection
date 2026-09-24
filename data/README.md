# Folder Data

Seluruh isi folder ini **tidak di-commit** (lihat `.gitignore`), kecuali berkas
ini dan `.gitkeep`. Foto wajah adalah data biometrik — data pribadi spesifik
menurut UU No. 27 Tahun 2022 Pasal 4 ayat (2). Rincian protokol ada di
`docs/PRD.md` §6 dan §11.

## Aturan

1. **Nama asli tidak pernah masuk ke folder proyek.** Setiap peserta hanya
   punya kode `S01`, `S02`, …. Pasangan nama ↔ kode cukup tercatat di formulir
   persetujuan kertas (atau pindaiannya), disimpan **di luar** folder proyek.
2. Tidak ada foto yang diambil sebelum formulir persetujuan ditandatangani.
3. Foto diambil lewat tool `capture` (Fase 2), bukan disalin manual, supaya
   nama berkas dan `metadata.csv` selalu konsisten.
4. Peserta yang menarik diri dihapus dengan `python -m pcdface forget S03` —
   termasuk foto multi-wajah yang memuat dirinya.
5. Data dihapus setelah nilai UTS keluar, atau setelah jurnal terbit bila
   peserta menyetujui publikasi.

## Struktur

```
data/
├── README.md
├── subjects.csv            diisi manual dari formulir persetujuan
├── metadata.csv            diisi otomatis oleh `capture`, satu baris per foto
├── annotations/boxes.json  kotak wajah manual (ground truth), dari `annotate`
├── crops/                  potongan kotak wajah, dari `crop`
└── raw/
    ├── jarak/S01/jarak_S01_150cm_normal_04.jpg
    ├── cahaya/S01/cahaya_S01_100cm_redup_02.jpg
    ├── pose/S01/…                                    (P1)
    ├── multi/F5/multi_F5_03.jpg
    └── kosong/kosong_07.jpg
```

Subfolder per subjek (`S01/`) dan per formasi (`F1/`…`F6/`) dibuat otomatis
oleh `capture`.

## `subjects.csv` — yang diisi manual

Satu baris per peserta, diisi dari formulir yang **sudah ditandatangani**:

| Kolom | Isi | Keterangan |
|---|---|---|
| `subject_id` | `S01` | Urut sesuai formulir |
| `consent_research` | `ya` | Bagian 5 formulir. Tanpa `ya`, peserta tidak boleh difoto |
| `consent_publication` | `ya` / `tidak` | Bagian 6 formulir. `tidak` → wajahnya diburamkan di setiap contoh gambar |
| `consent_date` | `2026-09-30` | Tanggal tanda tangan, format ISO |

Contoh:

```
subject_id,consent_research,consent_publication,consent_date
S01,ya,tidak,2026-09-30
S02,ya,ya,2026-09-30
```

## `metadata.csv` — diisi otomatis

Kolom mengikuti PRD §6.6, ditambah tiga kolom:

- `session` — kode sesi pengambilan (`capture --session`, bawaan tanggal hari itu). Data boleh
  diambil di beberapa pertemuan; aturannya di PRD §6.2.
- `subjects` — kode peserta pada foto multi-wajah, urut **kiri → kanan di
  citra** (mis. `S02;S05;S01`), sama urutannya dengan `positions_cm`. Dibutuhkan
  `forget` untuk menemukan foto multi-wajah yang memuat seorang peserta.
- `pose` — `frontal`, `kiri`, `kanan`, `menunduk`, `mendongak` untuk set pose
  (P1); kosong untuk set lain.
