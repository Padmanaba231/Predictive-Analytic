# Laporan Proyek Machine Learning - Putu Padmanaba

## Domain Proyek
### Latar belakang
![Water](https://github.com/Padmanaba231/Predictive-Analytic/blob/7f9e27ca397cb59ef501e8a81c9b9a271dd7fc9a/ML/IMG/download%20(4).jpg)
<br>
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
<br>
Dari grafik ini kita bisa melihat bahwa terdapat ketidakseimbangan data. Hal ini tentunya tidak baik karena dapat mengakibatkan bias terhadap model Machine Learning yang akan kita buat nantinya. Masalah ketidakseimbangan data ini akan diselesaikan pada bagian data preparation.
<br>
<br>
### Korelasi antar fitur
Kita akan menghitung korelasi antar fitur yang ada menggunakan bantuan metode heatmap correlation. Didapatkan hasil heatmap sebagai berikut:
<br>
![heatmap_correl](https://github.com/Padmanaba231/Predictive-Analytic/blob/0d5e37e67b8f9cd04ae5f2f344fc8736d802e2a0/ML/IMG/Screenshot%202024-02-22%20144950.png)
<br>
Berdasarkan gambar di atas, kita mendapatkan hasil dari setiap korelasi antar fitur yang kita miliki. Jika diperhatikan, setiap fitur yang kita miliki pada dataset ternyata memiliki korelasi yang rendah. Dari sini kita bisa berasumsi bahwa fitur-fitur yang kita miliki pada dataset tidak memiliki pengaruh yang kuat dalam menentukan air yang layak untuk dikonsumsi. Asumsi ini nanti kita buktikan pada tahapan evaluasi model.

# Data Preparation
### Menangani Missing Value
Seperti yang sudah dijelaskan sebelumnya, data kita memiliki beberapa missing value. Kita memiliki missing value pada fitur "ph", "Sulfate", "Trihalomethanes". Karena jumlah missing value yang tidak sedikit, kita tidak bisa asal meng-drop/menghapus missing value yang kita miliki untuk menghindari kehilangan informasi yang sebenarnya berguna untuk membangun model. Pada kasus ini, kita akan mengisi missing value dengan nilai rata-rata fitur pada setiap kelas.
<br>
<div><img src="https://github.com/Padmanaba231/Predictive-Analytic/blob/ca08cce50cd243455119df9f9b07cda5c08a07ee/ML/IMG/Screenshot%202024-02-22%20151351.png" width="600"/></div>
<br>

### Membagi dataset
Kita akan membagi dataset menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Kita menggunakan metode train_test_split dari liblary sklearn. Rasio dari pambagian dataset sebesar 75% untuk data latih dan 25% untuk data uji. Data latih digunakan untuk melatih model kita sementara data uji digunakan untuk mengevaluasi model kita. Perlu diperhatikan bahwa kita harus melakukan pembagian terlebih dahulu sebelum melakukan standarisasi. Hal ini dilakukan agar tidak terjadi kebocoran informasi pada data uji. Selain itu, ketika ingin menyeimbangkan persebaran data menggunakan metode oversampling, kita dapat menerapkan oversampling pada data latih saja.
<br>

### Balancing data menggunakan SMOTE
<br>
<div><img src="https://github.com/Padmanaba231/Predictive-Analytic/blob/8e6f181e97942945bd355c3284118a6722e4043a/ML/IMG/1_CeOd_Wbn7O6kpjSTKTIUog.png" width="600"/></div>
<br>
<br>
Seperti yang telah kita ketahui sebelumnya, data kita mengalami ketidakseimbangan. Kita akan menangani hal ini menggunakan metode oversampling dengan bantuan fitur SMOTE. Kita akan mencoba menyeimbangkan dataset dengan meningkatkan ukuran sampel langka. Daripada membuang sampel berlimpah, sampel langka baru dihasilkan dengan menggunakan fitur SMOTE (Sintetis Minoritas Sampling Teknik). Setelah menerapkan metode ini, data kita akan lebih seimbang yang diharapkan dapat meningkatkan kinerja model kita nantinya.

### Standarisasi
Algoritma machine learning cenderung memberikan hasil yang lebih baik dan konvergen lebih cepat ketika data memiliki skala yang seragam atau mendekati distribusi normal. Untuk mencapai ini, proses scaling dan standarisasi sangat membantu dalam mengubah bentuk fitur data sehingga lebih mudah dipahami dan diolah oleh algoritma. Kita akan memanfaatkan fungsi standarisasi yang dimiliki oleh liblary sklearn.

# Modeling
Proyek ini menggunakan 3 algoritma Machine Learning:
1. KNN (K-Nearest Neighbour)
2. SVC (Suport Vector Classification)
3. Random Forest

## KNN
KNN merupakan singkatan dari K-Nearest Neighbor,algoritma ini bekerja dengan prinsip bahwa objek/kelas yang mirip cenderung berada pada jarak yang dekat satu sama lain.Dengan kata lain, data yang memiliki karakteristik serupa akan cenderung saling bertetangga dalam ruang fitur. Hal inilah yang membuat KNN salah satu algoritma yang cocok untuk menyelesaikan kasus klasifikasi. 
#### Tahapan Kerja Umum KNN
+ Memilih jumlah tetangga terdekat (K) yang akan digunakan untuk memutuskan kelas suatu data baru. 
+ Hitung jarak antara data baru yang akan diprediksi dengan setiap titik data dalam set pelatihan. 
+ Menentukan K tetangga berdasarkan jarak terkecil. Ini merupakan data pelatihan yang memiliki nilai atribut paling mirip dengan data baru.
+ Lakukan prehitungan mayoritas di antara tetangga terpilih untuk menentukan kelas dari data baru. Artinya, kelas yang paling umum di antara K tetangga tersebut akan menjadi prediksi kelas untuk data baru.

Pada kasus proyek ini menggunakan <strong>n_neighbors = 30</strong> tetangga. Penentuan nilai <strong>K</strong> sangat berpengaruh pada kinerja model. Setelah mencoba berbagai nilai <strong>K</strong> didapatkan nilai <strong>K</strong> yang terbaik adalah sebesar 30. Untuk parameter yang lain, pada proyek ini menggunakan parameter default

#### Kelebihan & Kekurangan KNN
##### Kelebihan
Algoritma KNN relatif sederhana dan mudah dipahami. Konsep dasarnya cukup simple, yaitu mengambil mayoritas kelas dari tetangga terdekat. Algoritma KNN cocok untuk data non linear karena KNN dapat bekerja dengan baik pada data yang memiliki batas keputusan non-linier atau kompleks.
##### Kekurangan
KNN memerlukan penentuan nilai <strong>K</strong> yang tepat agar model dapat bekerja dengan baik. Nilai <strong>K</strong> yang terlalu kecil dapat menyebabkan model sensitif terhadap noise, sedangkan nilai <strong>K</strong> yang terlalu besar dapat menyebabkan model terlalu halus dan kurang responsif terhadap perubahan lokal dalam data.

## SVC
SVC sebenarnya termasuk kedalam algoritma SVM(Support Vector Machine). SVM merupakan model Machine Learning multifungsi yang dapat digunakan untuk menyelesaikan permasalahan klasifikasi, regresi, dan pendeteksian outlier. Algoritma Support Vector Machine (SVM) bertujuan untuk mengidentifikasi hyperplane optimal dalam ruang berdimensi-N (dengan N fitur) yang dapat efektif memisahkan titik-titik data input secara optimal. Pada kasus klasifikasi menggunakan SVC(Support Vector Classifier).

### Tahapan Kerja SVC secara umum
+ SVC berupaya menemukan hyperplane yang memisahkan dua atau lebih kelas secara optimal. Hyperplane ini memiliki sifat yang dapat memaksimalkan margin, yaitu jarak antara hyperplane dan titik-titik terdekat dari setiap kelas, yang disebut sebagai Support Vectors.
+ SVM berusaha untuk memaksimalkan margin ini, karena margin yang lebih besar memberikan tingkat kepercayaan yang lebih baik terhadap klasifikasi yang dilakukan oleh model.
+ Ketika data tidak dapat dipisahkan secara linier, SVC menggunakan konsep kernel untuk mentransformasikan data ke ruang fitur yang lebih tinggi. Kernel membantu model SVC menangani kasus-kasus di mana batas keputusan antara kelas tidak dapat dijelaskan secara linear dalam ruang fitur asli.
+ Setelah menemukan hyperplane yang optimal, SVC menggunakan batas keputusan untuk mengklasifikasikan data baru. Data yang berada di satu sisi hyperplane dianggap sebagai satu kelas, sedangkan data di sisi lainnya dianggap sebagai kelas yang berbeda.

Pada proyek ini menggunakan nilai parameter <strong>C</strong> sebesar 5. Parameter C pada model Support Vector Classifier (SVC) menentukan sejauh mana model ini akan memberikan toleransi terhadap kesalahan klasifikasi pada data pelatihan. Parameter ini disebut juga sebagai parameter penalti kesalahan (error penalty) atau parameter keberatan (regularization parameter). Parameter selain parameter <strong>C</strong> menggunakan parameter default SVC.


