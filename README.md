
```
# Analisis Sentimen Bahasa Indonesia

Proyek ini adalah aplikasi **analisis sentimen teks Bahasa Indonesia** dengan dua pendekatan:
1. **Lexicon-based** (berdasarkan daftar kata positif & negatif)
2. **Naive Bayes sederhana** (manual, tanpa library ML berat)

Aplikasi tersedia dalam dua bentuk:
- **Script Python** (`sentiment.py`) untuk analisis via terminal
- **Aplikasi web statis** (`sentiment_analyzer.html`) yang bisa langsung dibuka di browser

Cocok untuk tugas kuliah, eksperimen NLP dasar, atau demo konsep analisis sentimen.

---

## Struktur File

```

├── sentiment.py              # Program analisis sentimen berbasis Python
├── sentiment_analyzer.html   # Versi web (HTML + CSS + JavaScript)
└── README.md                 # Dokumentasi proyek

````

---

## Persyaratan Sistem

### Untuk Python
- Python **3.8 atau lebih baru**
- Library:
  - `nltk`

### Untuk Versi Web
- Browser modern (Chrome, Edge, Firefox)
- **Tidak perlu server**, cukup buka file HTML

---

## Instalasi (Python)

1. Clone repository atau download project:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo
````

2. Install dependensi:

   ```bash
   pip install nltk
   ```

3. Jalankan program:

   ```bash
   python sentiment.py
   ```

Saat pertama kali dijalankan, NLTK akan otomatis mengunduh resource yang dibutuhkan (`punkt` dan `stopwords`).

---

## Cara Penggunaan (Python)

1. Buka file `sentiment.py`
2. Ubah isi variabel `kalimat` di bagian **MAIN PROGRAM**:

   ```python
   kalimat = "Teks yang ingin kamu analisis"
   ```
3. Jalankan ulang program
4. Hasil akan ditampilkan di terminal:

   * Analisis per bagian kalimat
   * Probabilitas positif / negatif
   * Perbandingan Lexicon-based vs Naive Bayes
   * Kesimpulan akhir sentimen

Contoh output:

* POSITIF
* NEGATIF
* NETRAL / SEIMBANG

---

## Cara Penggunaan (Versi Web)

1. Buka file:

   ```
   sentiment_analyzer.html
   ```
2. Masukkan teks pada textarea
3. Klik tombol **Analisis Sentimen**
4. Hasil akan ditampilkan:

   * Sentimen tiap bagian kalimat
   * Grafik probabilitas
   * Ringkasan sentimen keseluruhan

Tidak perlu install apa pun. Praktis untuk demo atau presentasi.

---

## Cara Kerja Singkat

* **Kalimat dipecah** berdasarkan koma dan kata penghubung (dan, tapi, namun, dll)
* Setiap bagian dianalisis secara terpisah
* Sistem menghitung kemunculan kata positif & negatif
* Naive Bayes dilatih dari data contoh sederhana
* Hasil akhir ditentukan dari dominasi sentimen

Analoginya:
Seperti menilai cerita panjang dengan membaca per paragraf, lalu menyimpulkan suasana besarnya.

---

## Keterbatasan

* Dataset masih kecil dan manual
* Belum mendukung negasi kompleks (misal: *“tidak terlalu buruk”*)
* Belum memakai model ML modern (LSTM, Transformer, dsb)

Namun justru ini kelebihannya kalau tujuannya **belajar konsep dasar**.

---

## Pengembangan Lanjutan (Opsional)

* Menambah dataset training
* Integrasi Flask / FastAPI
* Simpan hasil analisis ke file
* Dukungan emoji & slang Bahasa Indonesia

---

## Lisensi

Proyek ini bebas digunakan untuk **pembelajaran dan pengembangan non-komersial**.
Silakan modifikasi sesuai kebutuhan.

---
