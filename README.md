# Laporan Proyek Machine Learning - Putu Padmanaba

## Domain Proyek
### Latar belakang
![Water](https://github.com/Padmanaba231/Predictive-Analytic/blob/7f9e27ca397cb59ef501e8a81c9b9a271dd7fc9a/ML/IMG/download%20(4).jpg)
Air merupakan kebutuhan utama umat manusia. Kehidupan kita tidak pernah lepas akan kebutuhan terhadap air. Terutama air bersih dan air yang layak untuk dikonsumsi. Oleh karena itu, Penyediaan air dan sanitasi yang baik, serta pengelolaan sumber daya air yang baik merupakan hal yang wajib dalam menjaga kualitas air agar bisa digunakan dengan aman oleh masyarakat. Jika tidak diperhatikan dengan benar dan teliti, air tersebut dapat terkontaminasi dan tentunya menyebabkan air tersebut tidak layak dikonsumsi. Air yang terkontaminasi dan sanitasi yang tidak memadai memfasilitasi penularan penyakit seperti kolera, diare, disentri, hepatitis A, tifoid, dan polio. Mereka yang tidak memiliki akses ke air bersih dan sanitasi menghadapi risiko kesehatan yang dapat dicegah. Sehingga pemantauan terhadap kualitas air sangatlah penting untuk terus dipantau. Kita dapat memanfaatkan pendekatan Machine Learning dalam membantu mengklasifikasikan antara air yang layak dikonsumsi dan yang tidak layak dikonsumsi.
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
