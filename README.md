# Analisis Dataset Tech Layoffs Q1 2026

Repository ini merupakan hasil pengerjaan dari tugas Literasi Data dan Inteligensi Aritifisial, berupa analisis terhadap gelombang PHK di industri teknologi global sepanjang Q1 2026 (Januari–Maret 2026).

---

## Dataset

Dataset mencakup 28 perusahaan dari berbagai sektor dan negara, dengan deskripsi data sebagai berikut:

| Kolom | Deskripsi |
|---|---|
| `company` | Nama perusahaan |
| `layoff_date` | Tanggal PHK diumumkan |
| `jobs_cut` | Jumlah pekerjaan yang dihilangkan |
| `pct_workforce_cut` | Persentase dari total karyawan |
| `sector` | Sektor industri |
| `country` | Negara asal |
| `ai_cited` | Apakah AI disebut sebagai alasan? |
| `simultaneous_ai_investment_bn` | Investasi AI bersamaan (miliar USD) |
| `stock_change_day_pct` | Perubahan harga saham hari pengumuman |
| `company_revenue_2025_bn` | Pendapatan perusahaan 2025 (miliar USD) |
| `roles_most_affected` | Peran yang paling banyak terdampak |
| `replacement_roles` | Peran pengganti yang direncanakan |

---

## Insight

### 1. Alasan Utama Tech Layoff

Dari total sekitar 101.000 pekerjaan yang hilang di Q1 2026, lebih dari separuhnya berasal dari perusahaan yang secara eksplisit menyebut AI sebagai alasan dalam pengumuman resmi mereka. Perusahaan seperti Oracle (30.000 PHK), Block (40% tenaga kerja), dan WiseTech Global (25% tenaga kerja) menyatakan bahwa sistem AI menggantikan peran manusia di bidang QA, customer support, hingga database administration.

### 2. Sektor Paling Terdampak

Sektor Enterprise Software mencatat angka PHK tertinggi dengan 31.600 pekerjaan, diikuti Social Media/AI dan E-Commerce/Cloud masing-masing 16.000. Hal tersebut terjadi karena sektor-sektor ini memang paling cepat mengadopsi otomasi dan sedang dalam fase restrukturisasi besar setelah tech boom pandemi.

### 3. Puncak Gelombang Layoff

Kalau dilihat dari time-series, Februari 2026 jelas jadi bulan paling ramai dengan total 37.725 PHK, yang naik drastis dari Januari yang hanya 6.500. Ini hampir sepenuhnya ditarik oleh pengumuman Oracle (30.000) dan Cisco (6.000). Maret turun ke 17.800, tapi masih jauh di atas level awal tahun. Secara mingguan, akselerasi terbesar terjadi di minggu pertama Februari.

### 4. Korelasi Layoff dan Investasi AI

Yang menarik (dan agak ironis) adalah korelasinya, dimana perusahaan yang paling banyak melakukan PHK justru yang paling besar investasi AI-nya secara bersamaan. Korelasi antar variabel ini tercatat di r = 0,56. Meta misalnya, mengumumkan rencana PHK 16.000 orang sambil menggelontorkan $115 miliar untuk infrastruktur AI. Amazon juga sama, dengan 16.000 PHK, tapi investasi AI $100 miliar. Ini menunjukkan bahwa yang terjadi bukan perusahaan sedang kesulitan, tapi memang ada pergantian jenis tenaga kerja yang dibutuhkan.

### 5. Respons Stock Market

Dari 28 kejadian PHK yang ada di dataset, 17 di antaranya direspons dengan kenaikan harga saham pada hari yang sama. Hanya 8 yang turun. Jadi investor tampaknya melihat PHK berbasis AI ini sebagai kabar baik, seperti sinyal bahwa perusahaan mau efisien dan serius berinvestasi ke teknologi baru.


### 6. Waktu Paling Umum Layoff

Distribusi dalam bulan menunjukkan pola yang cukup konsisten: pengumuman PHK paling banyak terjadi di tanggal 11–20 (total 53.858 pekerjaan), jauh lebih banyak dibanding awal bulan (36.238) atau akhir bulan (11.050). Kemungkinan ini berkaitan dengan siklus pelaporan keuangan kuartalan perusahaan publik.

---

## Kesimpulan

Secara keseluruhan, Q1 2026 memperlihatkan pola yang cukup jelas, dimana perusahaan-perusahaan besar sedang dalam proses mengganti jenis tenaga kerja yang mereka butuhkan, bukan sekadar memangkas biaya. PHK terbesar terjadi di peran-peran yang paling mudah diotomasi, sementara posisi yang berhubungan dengan AI justru dibuka. Yang jadi pertanyaan tentu saja seberapa cepat transisi ini akan terjadi di sektor-sektor lain yang belum masuk dataset ini.

---

## Tools

- Python (pandas, matplotlib, numpy)

---

## Sumber Dataset Kaggle
[Tech Layoffs Q1 2026](https://www.kaggle.com/datasets/alitaqishah/tech-layoffs-2026-ai-job-cuts-tracker)

---