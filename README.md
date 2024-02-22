# Laporan Proyek Machine Learning - Putu Padmanaba

## Domain Proyek
### Latar belakang
![Water](https://github.com/Padmanaba231/Predictive-Analytic/blob/7f9e27ca397cb59ef501e8a81c9b9a271dd7fc9a/ML/IMG/download%20(4).jpg)
Air merupakan kebutuhan utama umat manusia. Kehidupan kita tidak pernah lepas akan kebutuhan terhadap air. Terutama air bersih dan air yang layak untuk dikonsumsi. Oleh karena itu, Penyediaan air dan sanitasi yang baik, serta pengelolaan sumber daya air yang baik merupakan hal yang wajib dalam menjaga kualitas air agar bisa digunakan dengan aman oleh masyarakat. Jika tidak diperhatikan dengan benar dan teliti, air tersebut dapat terkontaminasi dan tentunya menyebabkan air tersebut tidak layak dikonsumsi. Air yang terkontaminasi dan sanitasi yang tidak memadai memfasilitasi penularan penyakit seperti kolera, diare, disentri, hepatitis A, tifoid, dan polio. Mereka yang tidak memiliki akses ke air bersih dan sanitasi menghadapi risiko kesehatan yang dapat dicegah. Sehingga pemantauan terhadap kualitas air sangatlah penting untuk terus dipantau. Kita dapat memanfaatkan pendekatan Machine Learning dalam membantu mengklasifikasikan antara air yang layak dikonsumsi dan yang tidak layak dikonsumsi.
<br>
Referensi: [Water quality classification using machine learning algorithms](https://www.sciencedirect.com/science/article/pii/S2214714422003646)

## Business Understanding
### Problem Statement
Berdasarkan latar belakang di atas, kita dapat menentukan pernyataan masalah sebagai berikut:
+ Bagaimana pengaruh fitur dalam menentukan kelayakan konsumsi air?
+ Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
+ Algoritma apa yang memiliki kinerja paling baik?

### Goals
+ Mengetahui pengaruh fitur dalam menentukan kelayakan konsusi air
+ Mengetahui cara pemrosesan data agar dapat dilatih dengan baik oleh model
+ Mengetahui model yang memiliki kinerja terbaik

### Solution Steatment
+ Menggunakan hubungan korelasi antar fitur untuk mengetahui pengaruh setiap fitur dalam menentukan kelayakan konsumsi air. Menggunakan hasil evaluasi model Machine Learning dalam menentukan pengaruh fitur dalam menentukan kelayakan konsumsi air.
+ Menerapkan beberapa metode dalam melakukan pemrosesan data seperti mengganti missing value dengan nilai rata-rata, membagi dataset menjadi data latih dan data pengujian, serta menerapkan upsampling ketika data mengalami ketidakseimbangan
+ Menggunakan lebih dari 1 model yang dapat menyelesaikan masalah klasifikasi. Algoritma yang dipakai adalah K-Nearest Neighbour, Random Forest, dan Suport Vector Classification

# Data Understanding
Dataset yang digunakan dalam proyek ini merupakan data yang berisikan beberapa parameter yang digunakan dalam menentukan kualitas air. Dataset ini dapat diunduh di [Kaggle: Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability/data)

Informasi dataset:
+ Dataset dalam format CSV (Comma-Seperated Values)
+ Dataset ini memiliki 10 fitur dengan 3276 sample
+ Data set memiliki 9 fitur bertipe float64 dan 1 fitur bertipe int64
+ Terdapat missing value pada dataset

### Variable pada dataset
+ pH value: Nilai pH air (tingkat keasaman suatu cairan)
+ Hardness: Kandungan mineral-mineral dalam air yaitu, ion kalsium (Ca) dan magnesium (Mg) dalam bentuk garam karbonat.
+ Solids: Total padatan terlarut dalam ppm(part per million)
+ Chloramines: Jumlah Kloramin dalam ppm(part per million)
+ Sulfate: Jumlah Sulfat yang dilarutkan dalam mg/L
+ Conductivity: Konduktivitas listrik air dalam μS/cm
+ Organic_carbon: Jumlah karbon organik dalam ppm(part per million)
+ Trihalomethanes: Jumlah Trihalometana dalam μg/L.
+ Turbidity: Ukuran sifat pemancar cahaya air di NTU(tingkat kekeruhan air)
+ Potability: Menunjukkan apakah air aman untuk dikonsumsi manusia. Dapat diminum 1 dan Tidak dapat diminum 0

#### Missing value
Kita akan menggunakan fungsi isnull().sum() untuk mengetahui jumlah missing value dan fitur yang memiliki missing value
![miss_val](https://github.com/Padmanaba231/Predictive-Analytic/blob/b36de2186af691f566a13d4f6db7203bfab67c4c/ML/IMG/Screenshot%202024-02-21%20173813.png)

### Exploratory Data Analys
#### Persebaran data
![persebaran data](https://github.com/Padmanaba231/Predictive-Analytic/blob/817f7bfa75f005b46b62e11afce5f3669fa59502/ML/IMG/Screenshot%202024-02-21%20174418.png)
<br>
Jika kita memperhatikan persebaran data pada gambar, persebaran data relatif seimbang kecuali pada fitur "Potability". Jika kita perhatikan lebih detail pada fitur "Potability" akan menampilkan grafik seperti berikut:
![grafik batang](https://github.com/Padmanaba231/Predictive-Analytic/blob/76936eba73623adb809da3c0622d5607f7f41801/ML/IMG/Screenshot%202024-02-22%20144156.png)









