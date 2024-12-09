# Laporan Proyek Machine Learning - Yudianto Sujana

## Domain Proyek

Penyakit jantung telah lama menjadi salah satu penyebab utama kematian global. Kompleksitas penyakit ini, yang dipengaruhi oleh berbagai faktor genetik, lingkungan, dan gaya hidup, telah menjadi tantangan besar dalam dunia medis. Diagnosis dini dan akurat sangat krusial untuk meningkatkan prognosis pasien. Namun, metode diagnosis tradisional seringkali terbatas dan membutuhkan waktu yang cukup lama.

Kecerdasan Buatan, khususnya Machine Learning, menawarkan harapan baru dalam mengatasi tantangan ini. Dengan kemampuannya dalam mengolah data dalam skala besar dan menemukan pola yang kompleks, Machine Learning dapat membantu dokter dalam membuat diagnosis yang lebih akurat dan cepat. Dalam konteks penyakit jantung, Machine Learning dapat digunakan untuk mengidentifikasi pola-pola tersembunyi dalam data pasien, seperti rekam medis, hasil tes laboratorium, dan pencitraan medis, yang dapat mengindikasikan risiko terjadinya penyakit jantung.

Beberapa penelitian telah menggunakan algoritma machine learning untuk prediksi penyakit jantung. Penelitian yang dilakukan oleh El-Sofany, Bouallegue & El-Latif (2024) membandingkan algoritma Support Vector Machines (SVM), XGBoost, bagging, Decision Trees (DT), and Random Rorests (RF) untuk memprediksi resiko penyakit jantung. Hasil penelitian menyimpulkan bahwa algoritma XGBoost memiliki performa yang lebih baik digandingkan algoritma lainnya, dengan akurasi sebesar 97,57% [1]. Penelitian lainnya yang dilakukan oleh Nashif, Raihan, Islam, & Imam (2018) menghasilkan kesimpulan bahwa algoritma SVM memiliki performa yang sangat baik dalam memprediksi resiko penyakit jantung dengan akurasi sebesar 97,53% [2]. Penelitian lainnya yang dilakukan oleh Garg, Sharma and Khan (2021) menggunakan algoritma K-Nearest Neighbor (K-NN) dan Random Forest dengan akurasi sebesar 86,88% untuk K-NN dan 81,97 untuk Random Forest [3]. Selain algoritma machine learning tradisional, algoritma Deep Neural Network atau Deepl Learning juga digunakan untuk memprediksi penyakit jantung. Penelitian oleh Bhatt, Patel, Ghetia, and Mazzeo (2023) menggunakan Multilayer Perceptron (MLP) menghasilkan akurasi sebesar 87,28% [4]. Penelitian lainnya oleh Honi & Szathmary (2024) menggunakan Convolutional Neural Network menghasilkan akurasi sebesar 99.95% [5].

Sejumlah kajian telah mengeksplorasi penerapan algoritma machine learning dalam prediksi penyakit jantung. El-Sofany, Bouallegue, & El-Latif (2024) melakukan perbandingan kinerja algoritma Support Vector Machines (SVM), XGBoost, bagging, Decision Trees (DT), dan Random Forests (RF) dalam memprediksi risiko penyakit jantung. Hasil penelitian menunjukkan bahwa algoritma XGBoost unggul dengan akurasi sebesar 97,57% [1]. Temuan lainnya yang dilaporkan oleh Nashif, Raihan, Islam, & Imam (2018) yang menyatakan bahwa algoritma SVM mampu mencapai akurasi sebesar 97,53% dalam prediksi risiko yang sama [2]. Studi komparatif oleh Garg, Sharma, dan Khan (2021) menunjukkan bahwa algoritma K-Nearest Neighbor (K-NN) dan Random Forest menghasilkan akurasi masing-masing sebesar 86,88% dan 81,97% [3].

Selain algoritma machine learning tradisional, algoritma Deep Neural Network (Deep Learning) juga telah diaplikasikan dalam domain ini. Bhatt, Patel, Ghetia, dan Mazzeo (2023) melaporkan bahwa model Multilayer Perceptron (MLP) mampu mencapai akurasi sebesar 87,28% dalam prediksi penyakit jantung [4]. Honi & Szathmary (2024) lebih lanjut menunjukkan potensi Convolutional Neural Network (CNN) dengan akurasi yang sangat tinggi, yaitu 99.95% [5]. Hasil-hasil penelitian ini mengindikasikan bahwa algoritma machine learning, baik tradisional maupun deep learning, memiliki potensi yang signifikan dalam mendukung diagnosis dini dan pengelolaan penyakit jantung.

Referensi:

[1] [A proposed technique for predicting heart disease using machine learning algorithms and an explainable AI method](https://www.nature.com/articles/s41598-024-74656-2)

[2] [Heart Disease Detection by Using Machine Learning Algorithms and a Real-Time Cardiovascular Health Monitoring System](https://www.scirp.org/journal/paperinformation?paperid=88650)

[3] [Heart disease prediction using machine learning techniques](https://iopscience.iop.org/article/10.1088/1757-899X/1022/1/012046)

[4] [Effective Heart Disease Prediction Using Machine Learning Techniques](https://www.mdpi.com/1999-4893/16/2/88)

[5] [A one-dimensional convolutional neural network-based deep learning approach for predicting cardiovascular diseases](https://www.sciencedirect.com/science/article/pii/S2352914824000911)

## Business Understanding

Penyakit jantung tetap menjadi salah satu penyebab utama kematian di seluruh dunia, dengan prevalensi yang terus meningkat seiring dengan faktor-faktor risiko yang melibatkan gaya hidup, usia, dan faktor genetik. Dalam konteks ini, upaya untuk mengidentifikasi individu yang berisiko tinggi terkena penyakit jantung menjadi sangat penting. Dengan semakin berkembangnya teknologi dan metode analisis data, algoritma machine learning (ML) telah muncul sebagai alat yang potensial untuk melakukan prediksi dini terhadap risiko penyakit jantung, memungkinkan intervensi yang lebih cepat dan efektif. Hal ini tidak hanya memiliki dampak signifikan pada pengurangan angka kematian, tetapi juga berpotensi mengurangi biaya perawatan kesehatan melalui pencegahan yang lebih baik.

Pendekatan berbasis machine learning menawarkan keuntungan dalam hal kemampuan untuk mengolah data dalam jumlah besar dan kompleksitas tinggi, serta menemukan pola yang mungkin tidak terdeteksi oleh metode analisis tradisional. Oleh karena itu, pemanfaatan machine learning dalam prediksi penyakit jantung berpotensi memberikan kontribusi besar dalam meningkatkan hasil kesehatan masyarakat.

### Problem Statements

- Banyak sistem tradisional yang digunakan untuk mendeteksi penyakit jantung, seperti pemeriksaan klinis berbasis gejala atau metode diagnostik manual, yang memiliki keterbatasan dalam akurasi dan kecepatan. Hal ini dapat menyebabkan diagnosis yang terlambat atau kesalahan dalam penentuan risiko pasien.

- Faktor-faktor yang mempengaruhi risiko penyakit jantung sangat beragam dan sering kali kompleks. Banyak individu mungkin memiliki kombinasi faktor risiko yang sulit terdeteksi dengan cara konvensional, seperti pola makan, riwayat keluarga, dan faktor genetik, yang dapat disarikan dengan lebih baik menggunakan pendekatan berbasis data.

- Dengan meningkatnya jumlah data medis yang tersedia, tantangan utama adalah bagaimana menganalisis data tersebut secara efisien dan menghasilkan prediksi yang akurat. Pendekatan berbasis machine learning dapat mengatasi masalah ini dengan mengidentifikasi pola-pola yang tidak terlihat oleh metode analisis tradisional.

### Goals

- Mengembangkan dan menerapkan model machine learning yang dapat memprediksi risiko penyakit jantung dengan akurasi yang lebih tinggi dibandingkan dengan metode tradisional. Ini termasuk mengidentifikasi faktor-faktor risiko utama dan mengukur dampaknya terhadap kesehatan pasien secara lebih tepat.

- Model yang dihasilkan dapat memberikan informasi yang lebih jelas dan berbasis data kepada pasien mengenai faktor-faktor risiko yang mempengaruhi kesehatan jantung mereka, serta mendorong perubahan perilaku yang dapat mencegah penyakit jantung.

- Model machine learning secara teratur dapat dioptimasi dengan mengelola dan menganalisis volume besar data medis dengan efisien, sehingga menghasilkan prediksi yang lebih akurat tentang kemungkinan risiko penyakit jantung.

### Solution statements

- Membandingkan beberapa algoritma machine learning untuk mendapatkan algoritma yang memiliki performa terbaik.
- Melakukan evaluasi menggunakan beberapa metrik seperti akurasi, presisi, F1, dan juga ROC agar model yang dihasilkan akurat dan dapat diandalkan.

## Data Understanding

Untuk memahami tentang dataset yang digunakan, teknik Exploratory Data Analysis (EDA) digunakan sebagai langkah awal dalam proses analisis data. EDA memungkinkan eksplorasi dan pengamatan terhadap dataset secara sistematis, sehingga karakteristik, pola, dan hubungan antara variabel-variabel dalam dataset dapat dipahami lebih jelas. Dengan menggunakan EDA, identifikasi outlier, deteksi pola dan hubungan antara variabel, serta pemahaman distribusi data dapat dilakukan, sehingga keputusan yang lebih tepat dalam proses analisis dan pengambilan keputusan berbasis data dapat diambil.

### Exploratory Data Analysis (EDA)

#### Deskripsi Variabel

Dataset yang digunakan berisi 1.888 data yang digabungkan dari lima dataset penyakit jantung yang tersedia. Dataset ini memiliki 14 fitur untuk memprediksi risiko serangan jantung, yang mencakup faktor medis dan demografi. Berikut ini adalah deskripsi terperinci dari setiap fitur.

1. age: Umur pasien (numerik).
2. sex: Jenis kelamin. Nilai: 1 = laki-laki dan 0 = perempuan.
3. cp: Jenis sakit di dada. Nilai: 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic.
4. trestbps: Tekanan darah saat istirahat (mm Hg) (numerik).
5. chol: Tingkat kolesterol serum (mg/dl) (numerik).
6. fbs: Gula darah puasa > 120 mg/dl. Nilai: 1 = true, 0 = false.
7. restecg: Hasil elektrokardiografi saat istirahat. Nilai: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy.
8. thalach: Denyut jantung maksimum (numerik).
9. exang: Angina akibat olahraga. Nilai: 1 = yes, 0 = no.
10. oldpeak: Depresi ST disebabkan oleh olahraga dibandingkan dengan istirahat (numerik).
11. slope: Kemiringan segmen ST latihan puncak. Nilai: 0 = Upsloping, 1 = Flat, 2 = Downsloping.
12. ca: Jumlah pembuluh darah utama (0-3) yang diwarnai dengan fluoroskopi. Nilai: 0, 1, 2, 3.
13. thal: Jenis-jenis talasemia. Nilai: 1 = Normal, 2 = Fixed defect, 3 = Reversible defect.
14. target: Label, resiko serangan jantung. Nilai: 1 = more chance of heart attack, 0 = less chance of heart attack.

Dari deskripsi fitur diatas, dapat dilihat bahwa terdapat 5 kolom bertipe numerik, 8 kolom bertipe kategori, dan 1 kolom target.

Berikut ini statistik dari dataset

![statistik](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/statistik.png)

Jumlah data untuk label 0 adalah 911, dan label 1 adalah 977, sehingga dapat dikatakan bahwa distribusi datanya seimbang.
![distribusi label](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/distribusi_label.png)

Dataset dapat diunduh di [Kaggle](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset).

#### Missing Values

Pada dataset ini tidak ditemukan adanya missing values, namun pada kolom oldpeak, terdapat 609 data yang yang bernilai 0. Untuk itu data tersebut dihapus sehingga jumlah data menjadi 1279 dengan rincian sebagai berikut:

- Jumlah data dengan label 0 adalah 714
- Jumlah data dengan label 1 adalah 565

Setelah penghapusan data birnilai 0, data masih dapat dikatakan seimbang.

### Outliers

Pada tahapan ini dilakukan analisa outliers menggunakan Boxplot. Berikut ini adalah hasil distribusi dan analisis outlier dari kolom-kolom bertipe numerik.

- age: Sebagian besar peserta memiliki usia antara 50-60 tahun. Tidak ada outlier yang signifikan.
- trestbps: Sebagian besar peserta memiliki tekanan darah istirahat sekitar 120-140 mmHg. Terdapat beberapa outlier dengan tekanan darah yang lebih tinggi.
- chol: Sebagian besar peserta memiliki kadar kolesterol sekitar 200-300 mg/dL. Terdapat beberapa outlier dengan kadar kolesterol yang lebih tinggi.
- thalachh: Sebagian besar peserta memiliki detak jantung maksimum sekitar 150-160 bpm. Terdapat beberapa outlier dengan detak jantung maksimum yang lebih rendah.
- oldpeak: Sebagian besar peserta memiliki depresi ST yang rendah (dekat dengan 0). Terdapat beberapa outlier dengan depresi ST yang lebih tinggi.

Berikut ini adalah boxplot dari setiap kolom bertipe numerik.
![boxplot](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/boxplot.png)

Setelah mengetahui adanya outliers, maka data outliers tersebut dihapus. Setelah penghapusan data, maka jumlah data menjadi 914 dengan rincian 523 untuk label 0 dan 391 untuk label 1, dan distribusi label masih seimbang.

### Univariate Analysis

#### Kolom kategorikal

Analisis univariate kolom kategorikal dilakukan menggunakan countplot. Hasilnya dapat dilihat pada grafik berikut ini.

![countplot](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/univariate_categorical.png)

Dari hasil analisis univariate dapat disimpulkan beberapa hal berikut ini:

1. Pada kolom sex, data terbanyak ada pada kategori laki-laki.
2. Pada kolom cp, jenis sakit di dada yang paling dominan adalah Typical angina.
3. Pada kolom fbs, seluruh data memiliki gula darah < 120 mg/dl.
4. Pada kolom restecg, sasil elektrokardiografi saat istirahat yang dominan adalah normal dan ST-T wave abnormality.
5. Pada kolom exang, data terbanyak adalah angina akibat olahraga.
6. Pada kolom slope, kemiringan segmen ST latihan puncak yang dominan adalah Flat.
7. Pada kolom ca, jumlah pembuluh darah utama (0-3) yang diwarnai dengan fluoroskopi paling banyak adalah 0.
8. Pada kolom thal, jenis talasemia yang dominan adalah Reversible defect dan fixed.

#### Kolom Numerik

Pada kolom bertipe numerik, analisis univariate dilakukan menggunakan histogram. Hasil dari histogram dapat dilihat dibawah ini.

![histogram](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/univariate_numerical.png)

Dari histogram diatas dapat disimpulkan beberapa hal:

1. Age: Distribusi cenderung normal dengan puncak di sekitar usia 55-60 tahun. Ini menunjukkan bahwa sebagian besar subjek penelitian berada dalam rentang usia tersebut. Sehingga dapat disimpulkan bahwa sebagian besar pasien dalam dataset ini berada pada usia menengah hingga lanjut.
2. Trestbps: Distribusi cenderung normal, dengan sedikit kemiringan ke kanan. Puncaknya berada di sekitar 120-130 mmHg. Sehingga dapat disimpulkan bahwa sebagian besar pasien memiliki tekanan darah dalam rentang normal hingga sedikit tinggi saat istirahat.
3. Chol: Distribusi cenderung normal, dengan sedikit kemiringan ke kanan. Puncaknya berada di sekitar 200-250 mg/dL. Sehingga dapat disimpulkan bahwa tingkat kolesterol sebagian besar pasien berada dalam rentang normal hingga sedikit tinggi.
4. Thalachh: distribusi cenderung normal, dengan sedikit kemiringan ke kiri. Puncaknya berada di sekitar 150-160 bpm. Sehingga dapat disimpulkan bahwa detak jantung maksimum sebagian besar pasien berada dalam rentang normal hingga sedikit di atas rata-rata.
5. Oldpeak: distribusi cenderung miring ke kanan, dengan sebagian besar data terkonsentrasi di sekitar nilai 0-1. Sehingga dapat disimpulkan bahwa sebagian besar pasien memiliki depresi segmen ST yang rendah.

### Multivariate Analysis

#### Categorical

Analisis multivariate pada kolom bertipe kategorikal menggunakan catplot. Berikut ini hasil dari catplot.

![catplot](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/catplot.png)

Dari catplot diatas, dapat dilihat bahwa terdapat beberapa fitur yang memiliki pengaruh ke target, yaitu: sex, exang, ca, dan thal.

#### Numerical

Analisis multivariate pada kolom bertipe numerik menggunakan matriks korelasi. Hasil analisis menunjukkan bahwa tidak ada satupun fitur yang memiliki korelasi kuat ke target.

Berikut ini adalah matriks korelasi dari kolom bertipe numerik.

![correlation matrix](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/cm.png)

## Data Preparation

### Encoding

Langkah pertama dalam tahapan data preparation adalah melakukan encoding terhadap fitur kategorikal. Encoding dilakukan untuk menghindari penafsiran yang salah oleh model dan meningkatkan kinerja model. Teknik encoding yang digunakan adalah One Hot Encoding.

Berikut ini struktur kolom setelah encoding.

![struktur encoding](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/struktur_encoding.png)

### Train test split

Langkah berikutnya adalah membagi dataset menjadi data train dan data test. Data train digunakan untuk training model machine learning, sedangkan data test digunakan untuk evaluasi model. 

Persentase pembagian data adalah sebesar 80% untuk data train dan 20% untuk data test, atau 731 data train dan 183 data test.

## Modeling

Untuk menentukan algoritma yang akan digunakan, maka dilakukan perbandingan beberapa algoritma machine learning untuk mengetahui algortima mana yang paling tepat untuk permasalahan deteksi penyakit jantung.

Algortima yang dibandingkan adalah:

1. Logistic Regression
2. Linear Discriminant Analysis
3. K-Nearest Neighbor(KNN)
4. Decision Tree
5. Naive Bayes
6. Support Vector Machine

Pemilihan algoritma-algoritma ini didasarkan pada karakteristik dan keunggulan masing-masing yang relevan dengan kasus prediksi penyakit jantung. berikut ini beberapa alasan pemilihan algoritma.

1. Logistic Regression cocok untuk masalah klasifikasi biner dan memberikan interpretasi yang jelas terhadap koefisien.
2. Linear Discriminant Analysis efektif dalam menangani masalah dengan fitur-fitur yang memiliki distribusi normal.
3. K-Nearest Neighbor (KNN) merupakan algoritma yang sederhana dan intuitif, serta dapat menangani data yang tidak linear.
4. Decision Tree memungkinkan interpretasi yang mudah dan dapat menangani fitur kategorikal.
5. Naive Bayes bekerja dengan baik pada dataset dengan ukuran besar dan fitur-fitur yang independen.
6. Support Vector Machine (SVM) efektif dalam menangani masalah klasifikasi dengan dimensi tinggi dan margin yang jelas.

Dengan membandingkan algoritma-algoritma tersebut, diharapkan dapat menemukan model yang paling akurat dan efisien untuk prediksi penyakit jantung.

Training dilakukan menggunakan teknik K-fold cross validation dengan n_split=10. Berikut ini adalah akurasi dan standar deviasi dari masing-masing algoritma.

Algoritma | Akurasi | Standard Deviasi
---------|----------|---------
Logistic Regression | 0.834598 | 0.048124
Linear Discriminant Analysis | 0.826472 | 0.062904
K-Nearest Neighbor(KNN) | 0.844113 | 0.039460
Decision Tree | 0.964421 | 0.012580
Naive Bayes | 0.816846 | 0.051139
Support Vector Machine | 0.915161 | 0.022826

Berdasarkan tabel diatas, dapat dilihat bahwa algoritma Decision Tree memiliki performa terbaik dibandingkan algoritma lainnya.

## Evaluation

Berdasarkan hasil perbandingan, maka dataset di training menggunkan algortima Decision Tree dengan parameter default. Setelah di training, model di evaluasi pada data test menggunakan beberapa matrix sebagai berikut:

### 1. Confusion Matrix

Confusion matrix adalah tabel yang digunakan untuk menggambarkan kinerja model klasifikasi. Confusion matrix biasanya terdiri dari empat komponen:

- **True Positive (TP):** Jumlah kasus yang sebenarnya positif (penderita penyakit jantung) dan diprediksi benar sebagai positif.
- **True Negative (TN):** Jumlah kasus yang sebenarnya negatif (bukan penderita penyakit jantung) dan diprediksi benar sebagai negatif.
- **False Positive (FP):** Jumlah kasus yang sebenarnya negatif (bukan penderita penyakit jantung) tetapi diprediksi salah sebagai positif (kesalahan tipe I).
- **False Negative (FN):** Jumlah kasus yang sebenarnya positif (penderita penyakit jantung) tetapi diprediksi salah sebagai negatif (kesalahan tipe II).

### 2. Akurasi (Accuracy)

Akurasi adalah proporsi prediksi yang benar dari total prediksi yang dibuat. Rumusnya adalah:

$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

Akurasi memberikan gambaran umum tentang seberapa baik model dalam memprediksi baik kasus positif maupun negatif. Namun, akurasi bisa menyesatkan jika dataset tidak seimbang (misalnya, banyak lebih banyak kasus negatif daripada positif).

### 3. Recall (Sensitivity atau True Positive Rate)

Recall mengukur seberapa baik model dapat mengidentifikasi kasus positif. Rumusnya adalah:

$\text{Recall} = \frac{TP}{TP + FN}$

Recall penting dalam kasus prediksi penyakit jantung karena mengurangi risiko false negative (menyatakan seseorang sehat padahal mereka memiliki penyakit jantung) sangat krusial.

### 4. Precision (Positive Predictive Value)

Precision mengukur seberapa baik model dapat memprediksi kasus positif tanpa membuat kesalahan. Rumusnya adalah:

$\text{Precision} = \frac{TP}{TP + FP}$

Precision penting untuk mengurangi false positive (menyatakan seseorang memiliki penyakit jantung padahal mereka sehat), yang juga bisa menimbulkan stres dan biaya tambahan untuk pasien.

### 5. F1 Score

F1 score adalah harmonik mean dari precision dan recall, yang memberikan satu nilai tunggal yang menggabungkan keduanya. Rumusnya adalah:

$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

F1 score berguna ketika kita ingin menyeimbangkan antara precision dan recall, terutama dalam kasus di mana false positive dan false negative memiliki dampak yang berbeda-beda.

Dalam prediksi penyakit jantung, penting untuk mempertimbangkan semua metrik ini karena setiap metrik memberikan informasi yang berbeda tentang kinerja model. Misalnya, recall penting untuk memastikan bahwa sebanyak mungkin kasus penyakit jantung terdeteksi, sementara precision penting untuk menghindari diagnosis palsu. F1 score memberikan gambaran yang seimbang antara keduanya. Confusion matrix dan akurasi memberikan gambaran umum, tetapi metrik-metrik lainnya membantu dalam evaluasi yang lebih detail dan berimbang.

### Hasil

Berikut ini hasil evaluasi dari model Decision Tree.

#### Confusion Matrix

Berikut ini hasil dari confusion matrix.

![confusion matrix](https://raw.githubusercontent.com/yest/proyek1-mlt/refs/heads/main/images/confusion_matrix.png)

- True Positive (TP): 74. Ini berarti model berhasil memprediksi 74 pasien sebagai "sakit jantung" dan prediksi tersebut benar.
- True Negative (TN): 105. Ini berarti model berhasil memprediksi 105 pasien sebagai "tidak sakit jantung" dan prediksi tersebut benar.
- False Positive (FP): 2. Ini berarti model salah memprediksi 2 pasien sebagai "sakit jantung" padahal sebenarnya mereka "tidak sakit jantung". (False alarm)
- False Negative (FN): 2. Ini berarti model salah memprediksi 2 pasien sebagai "tidak sakit jantung" padahal sebenarnya mereka "sakit jantung". (Missed detection)

Dapat disimpulkan bahwa model memiliki performa yang cukup baik dalam memprediksi penyakit jantung. Model ini mampu mengklasifikasikan sebagian besar pasien dengan benar. Namun, masih ada beberapa kasus di mana model melakukan kesalahan, yaitu mengklasifikasikan pasien yang sehat sebagai sakit (false positive) dan mengklasifikasikan pasien yang sakit sebagai sehat (false negative).

#### Akurasi, Recall, Precission, F1

Berikut ini hasi dari evaluasi.

Matriks | Nilai
---------|----------
Accuracy | 0.98
Recall | 0.98
Precission | 0.98
F1 | 0.98

Berdasarkan matriks diatas dapat disimpulkan bahwa model prediksi penyakit jantung memiliki performa yang sangat baik. Model ini sangat akurat dalam mengklasifikasikan pasien, baik yang sakit maupun yang sehat.
