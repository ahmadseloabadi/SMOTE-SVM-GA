# -*- coding: utf-8 -*-
"""preprosessing text.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17auzVw4u21yKiJYp16bb8PoQvUsEv9hT
"""

import pandas as pd
import numpy as np
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from sklearn import svm

"""# import dataset"""


#memasukan dataset

ulasan = pd.read_csv('data/Ulasan My XL 1000 Data Labelled.csv')
ulasan.head(10)

# Commented out IPython magic to ensure Python compatibility.
# melihat jumlah kelas sentimen
Jumlah_sentimen = ulasan['Sentimen'].value_counts()
print("jumlah sentimen :")
print(Jumlah_sentimen)
# menampilkan dalam bentuk plot diagram
import matplotlib.pyplot as plt
# %matplotlib inline
labels = ['negative' , 'neutral', 'positive']
plt.pie(ulasan.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
plt.legend()
plt.show()

"""# cleansing"""

def cleaningulasan(ulasan):
  ulasan = re.sub(r'@[A-Za-a0-9]+',' ',ulasan)
  ulasan = re.sub(r'#[A-Za-z0-9]+',' ',ulasan)
  ulasan = re.sub(r"http\S+",' ',ulasan)
  ulasan = re.sub(r'[0-9]+',' ',ulasan)
  ulasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", ulasan)
  #menghapus karakter tunggal
  ulasan = re.sub(r"\b[a-zA-Z]\b", " ", ulasan)
  ulasan = ulasan.strip(' ')
  return ulasan
ulasan['Cleaning']= ulasan['Ulasan'].apply(cleaningulasan)


def clearEmoji(ulasan):
    return ulasan.encode('ascii', 'ignore').decode('ascii')
ulasan['HapusEmoji']= ulasan['Cleaning'].apply(clearEmoji)

def replaceTOM(ulasan):
    pola = re.compile(r'(.)\1{2,}', re.DOTALL)
    return pola.sub(r'\1', ulasan)
ulasan['cleansing']= ulasan['HapusEmoji'].apply(replaceTOM)

ulasan[['Ulasan','cleansing']]
ulasan[['Ulasan','cleansing']].to_csv('cleansingb.csv', index=False,float_format='%.2f')

"""# case folding"""

def casefoldingText(ulasan):
  ulasan = ulasan.lower()
  return ulasan
ulasan['CaseFolding']= ulasan['cleansing'].apply(casefoldingText)
ulasan[['cleansing','CaseFolding']]
ulasan[['cleansing','CaseFolding']].to_csv('casefolding.csv', index=False)

"""# tokenizing"""

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
def tokenizingText(ulasan):
  ulasan = word_tokenize(ulasan)
  return ulasan
ulasan['Tokenizing']= ulasan['CaseFolding'].apply(tokenizingText)
ulasan[['CaseFolding','Tokenizing']]
ulasan[['CaseFolding','Tokenizing']].to_csv('tokenizing.csv', index=False)

"""# word normalization"""

def convertToSlangword(ulasan):
    kamusSlang = eval(open("drive/MyDrive/SKRIPSI/kodingan/slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in ulasan:
        filterSlang = pattern.sub(lambda x: kamusSlang[x.group()],kata)
        content.append(filterSlang.lower())
    ulasan = content
    return ulasan

ulasan['Formalisasi'] = ulasan['Tokenizing'].apply(convertToSlangword)
ulasan[['Tokenizing','Formalisasi']]
ulasan[['Tokenizing','Formalisasi']].to_csv('word normalization.csv', index=False)

"""# stopword removal"""

nltk.download('stopwords')
from nltk.corpus import stopwords

daftar_stopword = stopwords.words('indonesian')
# ---------------------------- manualy add stopword  ------------------------------------

# append additional stopword 
daftar_stopword.extend(["yg", "dg", "rt", "dgn", "ny", "d",'gb','ahk','g']) 

# read txt stopword using pandas 
txt_stopword = pd.read_csv("drive/MyDrive/SKRIPSI/kodingan/stopwords.txt", names= ["stopwords"], header = None)

# convert list to dictionary 
daftar_stopword = set(daftar_stopword)

# Mendefinisikan kata negasi
negation_words = ["tidak", "bukan", "tak", "tiada", "jangan", "gak"]

# Fungsi untuk menangani negasi
def handle_negation(tokens):
    result = []
    # negation = False
    for word in tokens:
        if word in negation_words:
            result.append("tidak_")
        elif word not in daftar_stopword:
            result.append(word)
    return result

def stopwordText(words):
 return [word for word in words if word not in daftar_stopword]

ulasan['Stopword Removal'] = ulasan['Formalisasi'].apply(handle_negation)
ulasan[['Formalisasi','Stopword Removal']]
ulasan[['Formalisasi','Stopword Removal']].to_csv('stopword.csv', index=False)

"""# stemming"""

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

factory = StemmerFactory()
stemmer = factory.create_stemmer()

#mengubah sebuah kalimat atau teks menjadi kata dasar
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}
#memasukan kata ke variable term_dict
for document in ulasan['Stopword Removal']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict)) 
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
print(term,":" ,term_dict[term])

print(term_dict) 
print("------------------------")
# Fungsi untuk mengubah kalimat menjadi kata dasar dan menangani negasi pada kalimat
def stemmingText(document):
    return [term_dict[term] for term in document]

ulasan['Stemming_list'] = ulasan['Stopword Removal'].apply(stemmingText)

#merubah list ke str 
ulasan['Stemming'] = ulasan['Stemming_list'].apply(' '.join)

#menampilkan data hasil stemming
ulasan[['Stopword Removal','Stemming']]
ulasan[['Stopword Removal','Stemming']].to_csv('stemming.csv', index=False)

ulasan.to_csv('dataset_bersih.csv',index=False)

"""# pembobotan tf-idf"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = ulasan['Stemming']
Y = ulasan['Sentimen']

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.25)

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)


# Create CountVectorizer instance
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(X)

# Create TfidfTransformer instance
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count)

# Create TfidfVectorizer instance
tfidf_vectorizer = TfidfVectorizer()
X_tfidf_vectorized = tfidf_vectorizer.fit_transform(X)

# Get the feature names from CountVectorizer or TfidfVectorizer
feature_names = count_vectorizer.get_feature_names_out()  # or tfidf_vectorizer.get_feature_names()

# Create a dictionary to store the results
results = {"Ulasan": [], "Term": [], "TF": [], "IDF": [], "TF-IDF": []}

# Loop over the documents
for i in range(len(X)):
    # Add the document to the results dictionary
    results["Ulasan"].extend([f" ulasan{i+1}"] * len(feature_names))
    # Add the feature names to the results dictionary
    results["Term"].extend(feature_names)
    # Calculate the TF, IDF, and TF-IDF for each feature in the document
    for j, feature in enumerate(feature_names):
        tf = X_count[i, j]
        idf = tfidf_transformer.idf_[j]  # or X_tfidf_vectorized.idf_[j]
        tf_idf_score = X_tfidf[i, j]  # or X_tfidf_vectorized[i, j]
        # Add the results to the dictionary
        results["TF"].append(tf)
        results["IDF"].append(idf)
        results["TF-IDF"].append(tf_idf_score)
# Convert the results dictionary to a Pandas dataframe
df = pd.DataFrame(results)


# Save the results to a CSV file
df.to_csv("tf_idf_results.csv", index=False)

#filter nilai term 
newdf = df[(df.TF != 0 )]
newdf
# Save the results to a CSV file
newdf.to_csv("hasil TF IDF.csv", index=False)

from sklearn.model_selection import KFold

x =x_train
y =y_train

kfold=5
kf = KFold(n_splits=kfold)
sum_of_error = 0
accuracies = []
for train_index,test_index in kf.split(x,y):
    x_train,x_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
        
    model = svm.SVC(kernel="rbf",C=5,gamma=0.1)
    model.fit(x_train,np.ravel(y_train))
        
    accuracy = model.score(x_test,y_test)
    accuracies.append(accuracy)

# Menampilkan plot akurasi
plt.plot(range(1, 6), accuracies, marker='o')
plt.xlabel('Fold')
plt.ylabel('Akurasi')
plt.title('Plot Akurasi K-Fold Cross Validation (k = 5)')
plt.ylim([0, 1])
plt.xticks(range(1, 6))
plt.grid(True)
plt.show()

"""# klasifikasi svm"""

from sklearn.metrics import confusion_matrix, accuracy_score 

# Making the SVM Classifer
Classifier = svm.SVC()

# Training the model on the training data and labels
Classifier.fit(x_train, y_train)

# Using the model to predict the labels of the test data
y_pred = Classifier.predict(x_test)

# import pickle
# pickle.dump(Classifier, open("svm_new", "wb"))
# Evaluating the accuracy of the model using the sklearn functions
accuracy = accuracy_score(y_test,y_pred)*100

clf = svm.SVC(class_weight='balanced')
clf.fit(x_train, y_train)

# Using the model to predict the labels of the test data
pred = clf.predict(x_test)

# import pickle
# pickle.dump(clf, open("svm_new", "wb"))
# Evaluating the accuracy of the model using the sklearn functions
acc = accuracy_score(y_test,pred)*100


# Printing the results
print("Accuracy for SVM is:",accuracy)
print("Accuracy for SVM balanced is:",acc)

# import seaborn as sns
# import matplotlib.pyplot as plt
# f, ax = plt.subplots(figsize=(8,5))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
# plt.xlabel("y_head")
# plt.ylabel("y_true")
# plt.show()
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
# cm_df = pd.DataFrame(conf_matrix,
#                      index = ['Positif','Netral','VIRGINICA'], 
#                      columns = ['SETOSA','VERSICOLR','VIRGINICA'])

"""# smote"""

x_res =vectorizer.fit_transform(ulasan['Stemming'])
y_res =ulasan['Sentimen']

from imblearn.over_sampling import SMOTE
from collections import Counter
import csv

print(f'Dataset sebelum SMOTE : {Counter(Y)}')

#penerapan smote
smote = SMOTE(sampling_strategy='auto',random_state=42)
X_smote, Y_smote = smote.fit_resample(x_res, y_res)

# Count original and synthetic data
counts = pd.DataFrame({'target': Y_smote}).value_counts().reset_index(name='counts')

df = pd.DataFrame(X_smote)
df.rename(columns={0:'term'}, inplace=True)
df['sentimen'] = Y_smote
# mengembalikan kalimat asli dari tfidf
feature_names = vectorizer.get_feature_names_out()
kalimat_asli = []
for index, row in df.iterrows():
    vektor_ulasan = X_smote[index]
    kata_kunci = [feature_names[i] for i in vektor_ulasan.indices]
    kalimat_asli.append(' '.join(kata_kunci))

# tambahkan kolom baru dengan kalimat asli ke dalam data frame
df['kalimat_asli'] = kalimat_asli
df.to_csv('data_smote.csv', index=False)
#mengambil data sintetik
df_sintetik = df.iloc[1000:]
#menyimpan dalam bentuk csv
df_sintetik.to_csv('data_sintetik.csv', index=False)

X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, test_size=0.20)

"""# svm+smote"""

# !pip install scipy
import scipy
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Making the SVM Classifer
classifier = svm.SVC()

# Training the model on the training data and labels
classifier.fit(X_train, Y_train)

# Using the model to predict the labels of the test data
Y_predsvm = classifier.predict(X_test)

# Evaluating the accuracy of the model using the sklearn functions
Accuracy = accuracy_score(Y_test,Y_predsvm)*100

# Printing the results
print("Accuracy for SVM is:",Accuracy)

print(classifier.get_params())

# data=scipy.sparse.csr_matrix.toarray(X_train)
# pca = PCA(n_components = 3).fit(data)
# X_pca = pca.transform(data)

data=scipy.sparse.csr_matrix.toarray(X_test)
pca = PCA(n_components = 2).fit(data)
X_pca = pca.transform(data)

# Tentukan parameter SVM
C = 1.0  # parameter regulasi
gamma = 0.7  # parameter kernel RBF

# Train model SVM
model = svm.SVC(kernel='rbf', gamma=gamma, C=C)
model.fit(X_pca, Y_test)

# Plot hasilnya
# buat meshgrid untuk plot
h = 0.02  # step size pada meshgrid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# buat plot titik data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_test, cmap=plt.cm.Set1,
            edgecolor='k')

# buat plot hyperplane dengan warna dan transparansi yang berbeda untuk setiap kelas
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.8)

# tambahkan label pada plot
plt.xlabel('Fitur 1')
plt.ylabel('Fitur 2')
plt.title('Hyperplane SVM RBF untuk Sentimen Ulasan Aplikasi')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import recall_score,precision_score
kfold = range(2, 12)
acc_svm = []
acc_gasvm = []

for k in kfold:
    acc_svm.append(np.mean(cross_val_score(classifier, X_train, Y_train, cv=k)))
    acc_gasvm.append(np.mean(cross_val_score(clf, X_train, Y_train, cv=k)))

plt.plot(kfold,  np.array(acc_svm)*100, '-o', label='SVM')
for x,y in zip(kfold, np.array(acc_svm)*100):
    plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom', fontsize=10)

plt.plot(kfold,  np.array(acc_gasvm)*100, '-o', label='GA-SVM')
for x,y in zip(kfold, np.array(acc_gasvm)*100):
    plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom', fontsize=10)

plt.xlabel('K-fold')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig('kfold_cross_validation.png')
plt.show()

# presision
prec_svm = []
prec_gasvm = []

for kf in kfold:
    y_pred_svm = cross_val_predict(classifier, X_train, Y_train, cv=kf)
    y_pred_gasvm = cross_val_predict(clf, X_train, Y_train, cv=kf)
    prec_svm.append(precision_score(Y_train, y_pred_svm, average='macro'))
    prec_gasvm.append(precision_score(Y_train, y_pred_gasvm, average='macro'))

plt.plot(kfold,  np.array(prec_svm)*100, '-o', label='SVM')
for x,y in zip(kfold, np.array(prec_svm)*100):
    plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom', fontsize=10)

plt.plot(kfold,  np.array(prec_gasvm)*100, '-o', label='GA-SVM')
for x,y in zip(kfold, np.array(prec_gasvm)*100):
    plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom', fontsize=10)

plt.xlabel('K-fold')
plt.ylabel('Precision')
plt.legend(loc='best')

plt.show()

# presision
rec_svm = []
rec_gasvm = []

for kfr in kfold:
    y_pred_svm = cross_val_predict(classifier, X_train, Y_train, cv=kfr)
    y_pred_gasvm = cross_val_predict(clf, X_train, Y_train, cv=kfr)
    rec_svm.append(recall_score(Y_train, y_pred_svm, average='macro'))
    rec_gasvm.append(recall_score(Y_train, y_pred_gasvm, average='macro'))
plt.plot(kfold,  np.array(rec_svm)*100, '-o', label='SVM')
for x,y in zip(kfold, np.array(rec_svm)*100):
    plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom', fontsize=10)

plt.plot(kfold,  np.array(rec_gasvm)*100, '-o', label='GA-SVM')
for x,y in zip(kfold, np.array(rec_gasvm)*100):
    plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom', fontsize=10)

plt.xlabel('K-fold')
plt.ylabel('recall')
plt.legend(loc='best')

plt.show()

from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score, classification_report
from sklearn.svm import SVC

c_values = [0.1, 1,5 ,10, 100]
gamma_values = [0.1, 1, 10, 100]
c=0.1

# inisialisasi list untuk menyimpan hasil
results = []

# ulangi model SVM dengan setiap nilai C dan gamma pada rentang yang ditentukan
for gamma in gamma_values:
    # inisialisasi model SVM dengan kernel RBF
    clasi = svm.SVC(kernel='rbf', C=c, gamma=gamma)

    # latih model dengan data training
    clasi.fit(X_train, Y_train)

    # prediksi label pada data testing
    y_pred = clasi.predict(X_test)

    # hitung metrik evaluasi model
    accuracy = accuracy_score(Y_test, y_pred)*100
    precision = precision_score(Y_test, y_pred, average='weighted')*100
    recall = recall_score(Y_test, y_pred, average='weighted')*100
    f1 = f1_score(Y_test, y_pred, average='weighted')*100
    # tambahkan hasil ke dalam list
    results.append({'c': c, 'gamma': gamma, 'accuracy': accuracy,'precision':"{:.2f}".format(precision), 'recall': recall, 'f1-score':"{:.2f}".format(f1) })

# simpan hasil ke dalam file CSV
df = pd.DataFrame(results)
df.to_csv('hasil_svm_C.csv', index=False)