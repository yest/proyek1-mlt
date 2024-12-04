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

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution statements

- Membandingkan beberapa algoritma machine learning untuk mendapatkan algoritma yang memiliki performa terbaik.
- Melakukan evaluasi menggunakan beberapa metrik seperti akurasi, presisi, F1, dan juga ROC agar model yang dihasilkan akurat dan dapat diandalkan.

## Data Understanding

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

Dataset dapat diunduh di [Kaggle](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset).

**Rubrik/Kriteria Tambahan (Opsional)**:

### Exploratory Data Analysis (EDA)

Berikut ini hasil dari (EDA)

1. Dari 1.888 data tidak ada data yang kosong.
2. Jumlah data untuk label 0 adalah 911, dan label 1 adalah 977, sehingga dapat dikatakan bahwa distribusi datanya seimbang.
images/distribusi_label.png
3. 


## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
