# Formulir Persetujuan Partisipasi Pengambilan Data Wajah

**Proyek:** Sistem Deteksi dan Pengenalan Wajah — Tugas UTS Mata Kuliah Pengolahan Citra Digital
**Versi formulir:** 4 (3 Oktober 2026) — menambahkan bagian 7, persetujuan pengenalan identitas
**Program Studi:** Teknik Informatika, Universitas Dian Nuswantoro
**Penanggung jawab:** ______________________________ (NIM ______________)
**Dosen pengampu:** ______________________________

---

## 1. Tujuan

Foto Anda akan dipakai untuk menguji seberapa baik beberapa program komputer menemukan **letak** wajah di dalam gambar pada berbagai jarak, kondisi cahaya, arah hadap kepala, ekspresi, wajah yang tertutup sebagian, dan jumlah orang, lalu memotong (*crop*) setiap wajah dari foto. Hasilnya dipakai untuk laporan tugas UTS dan, bila Anda menyetujui bagian 6, untuk artikel jurnal ilmiah.

**Hanya bila Anda menyetujui bagian 7**, foto Anda juga dipakai untuk menguji program **pengenalan wajah**: program mempelajari pola tekstur wajah Anda dari beberapa foto, lalu mencoba mencocokkan wajah di foto lain dengan **kode peserta** Anda (misalnya `S03`) — bukan dengan nama Anda. Bila Anda tidak menyetujui bagian 7, wajah Anda tidak pernah dipakai untuk pengenalan; program hanya menandai kotak di sekitar wajah Anda.

Sistem tidak pernah menampilkan atau menyimpan nama Anda, dan tidak menebak usia, jenis kelamin, emosi, suku, atau ras.

## 2. Data yang diambil

- Sekitar 135 foto wajah Anda sendirian melalui webcam laptop, pada jarak 0,5–3 meter, beberapa kondisi cahaya, beberapa arah hadap kepala (menoleh hingga ke samping, menunduk, mendongak, memiringkan kepala), beberapa ekspresi yang Anda peragakan (netral, senyum, marah, kaget), dan dengan wajah tertutup sebagian (masker, telapak tangan, kacamata hitam). Ekspresi dan penutup wajah hanya diminta sebagai variasi gambar; sistem tidak menilai atau menebak ekspresi, emosi, atau apa yang Anda kenakan.
- Sebagian foto Anda juga diolah menjadi versi yang sengaja dikaburkan (efek gerak) untuk menguji ketahanan program; tidak ada foto tambahan yang diambil untuk ini.
- Beberapa foto bersama anggota kelompok lain (2–4 orang dalam satu foto).
- Kotak lokasi wajah yang digambar secara manual pada foto-foto tersebut, dan potongan wajah (*crop*) dari foto-foto itu.
- Hanya bila Anda menyetujui bagian 7: **pola tekstur wajah** (histogram LBP) yang dihitung dari sekitar 14 foto Anda menghadap kamera. Pola ini adalah data biometrik yang bisa dipakai untuk mengenali wajah Anda, jadi diperlakukan sama ketatnya dengan foto.

Pengambilan foto dilakukan dalam satu atau beberapa pertemuan, sekitar 40 menit per orang ditambah sesi foto bersama.



Foto wajah termasuk data pribadi. Undang-Undang No. 27 Tahun 2022 tentang Pelindungan Data Pribadi, Pasal 4 ayat (2), menggolongkan data biometrik sebagai data pribadi yang bersifat spesifik.

## 3. Perlindungan data

- Anda diberi kode (misalnya `S03`). Nama asli Anda tidak dicantumkan pada nama berkas, data, kode program, maupun hasil penelitian.
- Foto disimpan hanya di laptop penanggung jawab, tidak diunggah ke repositori publik, media sosial, atau layanan penyimpanan awan.
- Foto dihapus setelah nilai UTS keluar. Bila Anda menyetujui publikasi di bagian 6, penghapusan dilakukan setelah artikel terbit.
- Pola tekstur wajah (bagian 7) disimpan hanya di laptop penanggung jawab, tidak pernah dibagikan, dan dihapus bersama foto Anda.

## 4. Hak Anda

- Partisipasi bersifat sukarela. Menolak tidak berpengaruh pada nilai atau hubungan Anda dengan kelompok.
- Anda boleh **menarik diri kapan saja** tanpa perlu memberi alasan. Seluruh foto Anda — termasuk foto bersama yang memuat wajah Anda — dan pola tekstur wajah Anda akan dihapus, dan Anda akan diberi tahu setelah penghapusan selesai.
- Anda boleh mencabut persetujuan bagian 7 saja tanpa menarik diri dari penelitian: pola tekstur wajah Anda dihapus dan wajah Anda tidak lagi dipakai untuk pengenalan.
- Anda boleh meminta melihat foto-foto Anda yang tersimpan.

## 5. Persetujuan penelitian

Dengan menandatangani formulir ini, saya menyatakan telah membaca dan memahami isi formulir, berkesempatan bertanya, dan **bersedia** difoto dan foto saya diolah untuk keperluan yang dijelaskan di atas.

## 6. Persetujuan publikasi wajah — pilih salah satu

Bagian ini terpisah. Anda tetap bisa ikut walaupun memilih tidak.

- [ ] **Ya** — foto wajah saya **boleh** ditampilkan sebagai contoh gambar di laporan UTS dan artikel jurnal.
- [ ] **Tidak** — foto wajah saya **tidak boleh** ditampilkan. Bila perlu contoh gambar, wajah saya harus diburamkan. Angka hasil penelitian yang tidak dapat dikaitkan dengan saya tetap boleh dilaporkan.

## 7. Persetujuan pengenalan identitas — pilih salah satu

Bagian ini terpisah. Anda tetap bisa ikut walaupun memilih tidak. **Peserta yang sudah menandatangani formulir versi sebelumnya juga perlu mengisi bagian ini** — formulir sebelumnya menyatakan sistem tidak mengenali siapa pun.

- [ ] **Ya** — wajah saya **boleh** dipakai untuk melatih dan menguji program pengenalan wajah dengan kode peserta saya, termasuk penyimpanan pola tekstur wajah seperti dijelaskan di bagian 2 dan 3.
- [ ] **Tidak** — wajah saya **tidak boleh** dipakai untuk pengenalan identitas. Foto saya hanya dipakai untuk deteksi dan *crop* wajah.

---

| | Peserta | Penanggung jawab |
|---|---|---|
| Nama | | |
| Kode peserta | | — |
| Tanda tangan | | |
| Tanggal | | |

---

*Formulir yang sudah ditandatangani berisi nama asli. Simpan dalam bentuk kertas atau pindaian di luar folder proyek — **jangan** masukkan ke repositori.*
