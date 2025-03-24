# SMOTE-SVM-GA

## Penerapan Optimasi Parameter SVM dengan Algoritma Genetika dan Teknik SMOTE

Proyek ini menerapkan optimasi parameter pada metode **Support Vector Machine (SVM)** menggunakan **Algoritma Genetika (GA)** serta teknik **Synthetic Minority Over-sampling Technique (SMOTE)** untuk menangani **imbalance dataset** dalam analisis sentimen. Aplikasi ini dibangun menggunakan **Streamlit** sebagai antarmuka visual.

## 📌 Fitur Aplikasi

### 🔹 **1. Halaman Home**

Menampilkan halaman utama aplikasi.

![Home](https://github.com/ahmadseloabadi/SMOTE-SVM-GA/assets/50831996/b6f8c262-c4a9-47ae-885d-3bcc4648791f)

### 🔹 **2. Halaman Pengolahan Data**

Menampilkan dataset, hasil **text preprocessing**, dan penerapan teknik **SMOTE** untuk menangani ketidakseimbangan data.

![Pengolahan Data](https://github.com/ahmadseloabadi/SMOTE-SVM-GA/assets/50831996/74e58ee2-c8a1-4a2a-9e90-cc30ea9a13e4)

### 🔹 **3. Halaman Algoritma Genetika**

Menampilkan hasil pengujian **Algoritma Genetika (GA)** serta menyediakan fitur untuk melakukan eksperimen penerapan **GA-SVM**.

![Algoritma Genetika](https://github.com/ahmadseloabadi/SMOTE-SVM-GA/assets/50831996/3a860bbc-baec-4dc0-b924-18ccdd9fa10c)

### 🔹 **4. Halaman Pengujian**

Menampilkan hasil pengujian pada kalimat baru menggunakan metode **SVM** dan **GA-SVM**.

![Pengujian](https://github.com/ahmadseloabadi/SMOTE-SVM-GA/assets/50831996/8ef6641c-bf0f-4d54-9350-9e29cf7fd276)

### 🔹 **5. Halaman Report**

Menampilkan evaluasi model menggunakan **k-fold cross-validation** dan **confusion matrix** pada metode **SVM** dan **GA-SVM**.

![Report](https://github.com/ahmadseloabadi/SMOTE-SVM-GA/assets/50831996/30943eb8-c0d1-409f-907a-c77af0477e87)

## 📦 Instalasi dan Penggunaan

### **1. Clone Repository**

```bash
git clone https://github.com/ahmadseloabadi/SMOTE-SVM-GA.git
cd SMOTE-SVM-GA
```

### **2. Buat dan Aktifkan Virtual Environment**

```bash
python -m venv myenv  # Membuat virtual environment
source venv/bin/activate  # Untuk Mac/Linux
venv\Scripts\activate  # Untuk Windows
```

### **3. Instal Dependensi**

```bash
pip install -r requirements.txt
```

### **4. Jalankan Aplikasi Streamlit**

```bash
streamlit run app.py
```

## 🔗 Referensi

- **Code Algoritma Genetika:** [SVM Optimization by Genetic Algorithm](https://github.com/kevingeorge0123/SVM-Opt-by-Genetic-Algorithm/tree/main)
- **Dataset Kaggle:** [Klik di sini](https://www.kaggle.com/datasets/dimasdiandraa/data-ulasan-terlabel)

## 📌 Lisensi

Proyek ini dikembangkan untuk keperluan penelitian dan pembelajaran. Silakan gunakan dengan bijak. 🚀
