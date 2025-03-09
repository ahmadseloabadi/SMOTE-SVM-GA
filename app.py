#import library
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import random as rd
import seaborn as sns

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import recall_score,precision_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

import Genetic_Algorithm as svm_hp_opt


# Set page layout and title
st.set_page_config(page_title="MyXL", page_icon="style/icon.png")

# Add custom CSS
def add_css(file):
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

add_css("style/style.css")

import pickle
#import dataset

ulasan = pd.read_csv('data/dataset_bersih.csv')

# text preprosessing
def cleansing(kalimat_baru): 
    kalimat_baru = re.sub(r'@[A-Za-a0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r'#[A-Za-z0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"http\S+",' ',kalimat_baru)
    kalimat_baru = re.sub(r'[0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", kalimat_baru)
    kalimat_baru = re.sub(r"\b[a-zA-Z]\b", " ", kalimat_baru)
    kalimat_baru = kalimat_baru.strip(' ')
    # menghilangkan emoji
    def clearEmoji(ulasan):
        return ulasan.encode('ascii', 'ignore').decode('ascii')
    kalimat_baru =clearEmoji(kalimat_baru)
    def replaceTOM(ulasan):
        pola = re.compile(r'(.)\1{2,}', re.DOTALL)
        return pola.sub(r'\1', ulasan)
    kalimat_baru=replaceTOM(kalimat_baru)
    return kalimat_baru
def casefolding(kalimat_baru):
    kalimat_baru = kalimat_baru.lower()
    return kalimat_baru
def tokenizing(kalimat_baru):
    kalimat_baru = word_tokenize(kalimat_baru)
    return kalimat_baru
def slangword (kalimat_baru):
    kamusSlang = eval(open("data/slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in kalimat_baru:
        filter_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata.lower())
        if filter_slang.startswith('tidak_'):
          kata_depan = 'tidak_'
          kata_belakang = kata[6:]
          kata_belakang_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata_belakang.lower())
          kata_hasil = kata_depan + kata_belakang_slang
          content.append(kata_hasil)
        else:
          content.append(filter_slang)
    kalimat_baru = content
    return kalimat_baru
def handle_negation(kalimat_baru):
    negation_words = ["tidak", "bukan", "tak", "tiada", "jangan", "gak",'ga']
    new_words = []
    prev_word_is_negation = False
    for word in kalimat_baru:
        if word in negation_words:
            new_words.append("tidak_")
            prev_word_is_negation = True
        elif prev_word_is_negation:
            new_words[-1] += word
            prev_word_is_negation = False
        else:
            new_words.append(word)
    return new_words
def stopword (kalimat_baru):
    daftar_stopword = stopwords.words('indonesian')
    daftar_stopword.extend(["yg", "dg", "rt", "dgn", "ny", "d",'gb','ahk','g','anjing','ga','gua','nder']) 
    # Membaca file teks stopword menggunakan pandas
    txt_stopword = pd.read_csv("data/stopwords.txt", names=["stopwords"], header=None)

    # Menggabungkan daftar stopword dari NLTK dengan daftar stopword dari file teks
    daftar_stopword.extend(txt_stopword['stopwords'].tolist())

    # Mengubah daftar stopword menjadi set untuk pencarian yang lebih efisien
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        cleaned_words = []
        for word in words:
            # Memisahkan kata dengan tambahan "tidak_"
            if word.startswith("tidak_"):
                cleaned_words.append(word[:5])
                cleaned_words.append(word[6:])
            elif word not in daftar_stopword:
                cleaned_words.append(word)
        return cleaned_words
    kalimat_baru = stopwordText(kalimat_baru)
    return kalimat_baru 
def stemming(kalimat_baru):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # Lakukan stemming pada setiap kata
    stemmed_words = [stemmer.stem(word) for word in kalimat_baru]
    return stemmed_words
vectorizer = TfidfVectorizer()

xd =vectorizer.fit_transform(ulasan['Stemming'])
yd =ulasan['Sentimen']
sx_train, sx_test, sy_train, sy_test = train_test_split(xd, yd, test_size=0.20)
x_res =vectorizer.fit_transform(ulasan['Stemming'])
y_res =ulasan['Sentimen']
#penerapan smote
smote = SMOTE(sampling_strategy='auto')
X_smote, Y_smote = smote.fit_resample(x_res, y_res)
X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, test_size=0.20)

classifier = svm.SVC()
classifier.fit(X_train, Y_train)

svmga=svm.SVC(kernel='rbf',C=12.7,gamma=0.3)
svmga.fit(X_train, Y_train)


kfold=5
#side bar
with st.sidebar :
    selected = option_menu('sentimen analisis',['Home','Pengolahan data','Algoritma Genetika','Pengujian','Report'])

if(selected == 'Home') :
    st.title('OPTIMASI PARAMETER METODE SUPPORT VECTOR MACHINE DENGAN ALGORITMA GENETIKA PADA ULASAN APLIKASI MyXL ')
    st.write('MyXL merupakan aplikasi self- service yang diberikan oleh PT XL Axiata Tbk pada Google Play Store yang berguna dalam proses memudahkan pengguna dalam melakukan layanan XL')
    image = Image.open('style/myxl-logo.png')
    st.image(image)



elif(selected == 'Pengolahan data') :
    tab1,tab2,tab3 =st.tabs(['Dataset','Text preprosesing','SMOTE'])
    with tab1 :
        st.title('dataset ulasan aplikasi my xl')
        st.text('dataset ulasan aplikasi MyXL yang di ambil dari Kaggle')
        st.write('link [kaggle](https://www.kaggle.com/datasets/dimasdiandraa/data-ulasan-terlabel?resource=download)')

        def filter_sentiment(dataset, Sentimen):
            return dataset[dataset['Sentimen'].isin(selected_sentiment)]
        sentiment_map = {1: 'positif', -1: 'negatif', 0: 'netral'}
        dataset =pd.read_csv('data/Ulasan My XL 1000 Data Labelled.csv')
        selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
        selected_sentiment = [key for key, value in sentiment_map.items() if value in selected_sentiment]
        filtered_data = filter_sentiment(dataset, selected_sentiment)
        st.dataframe(filtered_data)

        # Hitung jumlah kelas dataset
        st.write("Jumlah kelas:  ")
        kelas_sentimen = ulasan['Sentimen'].value_counts()
        datpos, datneg, datnet = st.columns(3)
        with datpos:
            st.markdown("Positif")
            st.markdown(f"<h1 style='text-align: center; color: green;'>{kelas_sentimen[1]}</h1>", unsafe_allow_html=True)
        with datnet:
            st.markdown("Netral")
            st.markdown(f"<h1 style='text-align: center; color: orange;'>{kelas_sentimen[0]}</h1>", unsafe_allow_html=True)
        with datneg:
            st.markdown("Negatif")
            st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen[-1]}</h1>", unsafe_allow_html=True)
        #membuat diagram
        labels = ['negatif' , 'neutral', 'positif']
        fig1,ax1=plt.subplots()
        ax1.pie(ulasan.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
        ax1.axis('equal')
        st.pyplot(fig1)
        st.write('masukan dataset :')

    with tab2 :
        opsi_prepro = st.selectbox('option',('statis', 'dinamis'))
        prepro=False
        if(opsi_prepro == 'statis') :
            st.title('Text preprosesing')
            st.header('cleansing')#----------------
            st.text('membersihkan data dari angka ,tanda baca,dll.')
            cleansing = pd.read_csv('data/cleansing.csv')
            st.write(cleansing)
            st.header('casefolding')#----------------
            st.text('mengubahan seluruh huruf menjadi kecil (lowercase) yang ada pada dokumen.')
            casefolding = pd.read_csv('data/casefolding.csv')
            st.write(casefolding)
            st.header('tokenizing')#----------------
            st.text('menguraikan kalimat menjadi token-token atau kata-kata.')
            tokenizing = pd.read_csv('data/tokenizing.csv')
            st.write(tokenizing)
            st.header('word normalization')#----------------
            st.text('mengubah penggunaan kata tidak baku menjadi baku')
            word_normalization = pd.read_csv('data/word normalization.csv')
            st.write(word_normalization)
            st.header('stopword')#----------------
            st.text('menyeleksi kata yang tidak penting dan menghapus kata tersebut.')
            stopword = pd.read_csv('data/stopword.csv')
            st.write(stopword)
            st.header('stemming')#----------------
            st.text(' merubahan kata yang berimbuhan menjadi kata dasar. ')
            stemming = pd.read_csv('data/stemming.csv')
            st.write(stemming)
            st.title('Pembobotan TF-IDF')
            st.text('pembobotan pada penelitan ini menggunakan tf-idf')
            tfidf = pd.read_csv('data/hasil TF IDF.csv')
            st.dataframe(tfidf,use_container_width=True)

        if(opsi_prepro == 'dinamis') :
            uploaded_file = st.file_uploader("Choose a file")

            if uploaded_file is not None:

                dataset = pd.read_csv(uploaded_file)
                st.write(dataset)

                dataset['Cleansing']= dataset['Ulasan'].apply(cleansing)
                st.write('hasil cleansing')
                cleansing = dataset[['Ulasan','Cleansing']]
                st.dataframe(cleansing)

                dataset['CaseFolding']= dataset['Cleansing'].apply(casefolding)
                st.write('hasil casefolding')
                casefolding= dataset[['Cleansing','CaseFolding']]
                st.dataframe(casefolding)
                
                dataset['Tokenizing']= dataset['CaseFolding'].apply(tokenizing)
                st.write('hasil tokenizing')
                tokenizing= dataset[['CaseFolding','Tokenizing']]
                st.dataframe(tokenizing)
                
                dataset['stemming']= dataset['Tokenizing'].apply(stemming)
                st.write('hasil stemming')
                stemming= dataset[['Tokenizing','stemming']]
                st.dataframe(stemming)

                dataset['negasi']= dataset['stemming'].apply(handle_negation)
                st.write('hasil negasi')
                negasi= dataset[['stemming','negasi']]
                st.dataframe(negasi)

                dataset['wordnormalization']= dataset['negasi'].apply(slangword)
                st.write('hasil wordnormalization')
                wordnormalization= dataset[['negasi','wordnormalization']]
                st.dataframe(wordnormalization)

                dataset['stopword']= dataset['wordnormalization'].apply(stopword)
                st.write('hasil stopword')
                stopword= dataset[['wordnormalization','stopword']]
                #merubah list ke str 
                dataset['hasilprepro'] = dataset['stopword'].apply(' '.join)
                st.dataframe(stopword)
                prepro=True

    with tab3 :
        menu_smote = st.selectbox('menu',('smote statis', 'smote dinamis'))
        if(menu_smote == 'smote statis') :
            st.title('SMOTE')
            st.text('SMOTE adalah teknik untuk mengatasi ketidak seimbangan kelas pada dataset')
            
            seb_smote,ses_smote = st.columns(2)
            with seb_smote:
                st.header('sebelum SMOTE')
                st.write('Jumlah dataset:',len(y_res))
                # Hitung jumlah kelas sebelum SMOTE
                st.write("Jumlah kelas:  ")
                Jum_sentimen = ulasan['Sentimen'].value_counts()
                spos, sneg, snet = st.columns(3)
                with spos:
                    st.markdown("Positif")
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{Jum_sentimen[1]}</h1>", unsafe_allow_html=True)
                with snet:
                    st.markdown("Netral")
                    st.markdown(f"<h1 style='text-align: center; color: orange;'>{Jum_sentimen[0]}</h1>", unsafe_allow_html=True)
                with sneg:
                    st.markdown("Negatif")
                    st.markdown(f"<h1 style='text-align: center; color: blue;'>{Jum_sentimen[-1]}</h1>", unsafe_allow_html=True)
                
                #membuat diagram
                labels = ['negatif' , 'neutral', 'positif']
                fig1,ax1=plt.subplots()
                ax1.pie(ulasan.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
                ax1.axis('equal')
                st.pyplot(fig1)

            with ses_smote:
                st.header('sesudah SMOTE')
                df = pd.DataFrame(X_smote)
                df.rename(columns={0:'term'}, inplace=True)
                df['sentimen'] = Y_smote
                #melihat banyak dataset
                st.write('Jumlah dataset :',len(Y_smote))
                # melihat jumlah kelas sentimen aetelah SMOTE
                st.write("Jumlah kelas: ")
                Jumlah_sentimen = df['sentimen'].value_counts()
                pos, neg, net = st.columns(3)
                with pos:
                    st.markdown("Positif")
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{Jumlah_sentimen[1]}</h1>", unsafe_allow_html=True)
                with net:
                    st.markdown("Netral")
                    st.markdown(f"<h1 style='text-align: center; color: orange;'>{Jumlah_sentimen[0]}</h1>", unsafe_allow_html=True)
                with neg:
                    st.markdown("Negatif")
                    st.markdown(f"<h1 style='text-align: center; color: blue;'>{Jumlah_sentimen[-1]}</h1>", unsafe_allow_html=True)
                
                # menampilkan dalam bentuk plot diagram
                labels = ['negatif' , 'neutral', 'positif']
                fig2,ax2=plt.subplots()
                plt.pie(df.groupby('sentimen')['sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
                ax2.axis('equal')
                st.pyplot(fig2)
            sintetis = pd.read_csv('data/data_sintetik.csv')
            dsmote = pd.read_csv('data/data_smote.csv')
            st.header('Data sintetis')
            st.write('jumlah data sintetis :',len(sintetis))
            st.write('jumlah penambahan setiap kelas :')
            sinpos,sinneg,sinnet = st.columns(3)
            with sinpos:
                st.markdown("Positif")
                sel_pos=Jumlah_sentimen[1]-Jum_sentimen[1]
                st.markdown(f"<h1 style='text-align: center; color: green;'>{sel_pos}</h1>", unsafe_allow_html=True)
            with sinnet:
                st.markdown("Netral")
                sel_net=Jumlah_sentimen[0]-Jum_sentimen[0]
                st.markdown(f"<h1 style='text-align: center; color: orange;'>{sel_net}</h1>", unsafe_allow_html=True)
            with sinneg:
                st.markdown("Negatif")
                sel_neg=Jumlah_sentimen[-1]-Jum_sentimen[-1]
                st.markdown(f"<h1 style='text-align: center; color: blue;'>{sel_neg}</h1>", unsafe_allow_html=True)
            def sentimen_sintetis(dataset, Sentimen):
                return sintetis[sintetis['sentimen'].isin(selected_sentiment)]
            def sentimen_smote(dataset, Sentimen):
                return dsmote[dsmote['sentimen'].isin(sentiment)]
            
            optiondata = st.selectbox('pilih data',('SMOTE', 'SINTETIS'))

            if(optiondata == 'SMOTE') :
                # Panggil fungsi pencarian
                st.write('menampilkan data sesudah',optiondata)
                sentiment_map = {1: 'positif', 0: 'netral',-1:'negatif'}
                sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
                sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
                filtered_data = sentimen_smote(dsmote, sentiment)
                st.dataframe(filtered_data)
            if(optiondata == 'SINTETIS') :
                st.write('menampilkan data ',optiondata)
                sentiment_map = {1: 'positif', 0: 'netral'}
                selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
                selected_sentiment = [key for key, value in sentiment_map.items() if value in selected_sentiment]
                filtered_data = sentimen_sintetis(sintetis, selected_sentiment)
                st.dataframe(filtered_data)
            with st.expander('---') :
                st.text('cari kalimat ')
                search_query = st.text_input("Masukkan kalimat yang ingin dicari:")
                search_results = dsmote[dsmote['kalimat_asli'].str.contains(search_query)]
                if st.button('cari') :
                    st.write(search_results)
                # Mencari baris duplikat berdasarkan nilai kalimat asli
                duplicates = dsmote[dsmote.duplicated(subset='kalimat_asli', keep=False)]
                results = {"index": [], "term": [], "sentimen": [], "kalimat_asli": []}
                # Menampilkan kalimat yang duplikat
                for index, row in duplicates.iterrows():
                    results["index"].append(index)
                    results["term"].append(row['term'])
                    results["sentimen"].append(row['sentimen'])
                    results["kalimat_asli"].append(row['kalimat_asli'])

                duplikat = pd.DataFrame(results)
                st.dataframe(duplikat)
        if(menu_smote == 'smote dinamis') :
            
            if prepro == True :
                x=vectorizer.fit_transform(dataset['hasilprepro'])
                y=dataset['Sentimen']    
                smote = SMOTE(sampling_strategy='auto')
                Xsmote, Ysmote = smote.fit_resample(x, y)
                sebelum_smote,sesudah_smote = st.columns(2)
                with sebelum_smote:
                    st.header('sebelum SMOTE')
                    st.write('Jumlah dataset:',len(y))
                    # Hitung jumlah kelas sebelum SMOTE
                    st.write("Jumlah kelas:  ")
                    Jum_sentimen = dataset['Sentimen'].value_counts()
                    spos, sneg, snet = st.columns(3)
                    with spos:
                        st.markdown("Positif")
                        st.markdown(f"<h1 style='text-align: center; color: green;'>{Jum_sentimen[1]}</h1>", unsafe_allow_html=True)
                    with snet:
                        st.markdown("Netral")
                        st.markdown(f"<h1 style='text-align: center; color: orange;'>{Jum_sentimen[0]}</h1>", unsafe_allow_html=True)
                    with sneg:
                        st.markdown("Negatif")
                        st.markdown(f"<h1 style='text-align: center; color: blue;'>{Jum_sentimen[-1]}</h1>", unsafe_allow_html=True)
                    
                    #membuat diagram
                    labels = ['negatif' , 'neutral', 'positif']
                    fig1,ax1=plt.subplots()
                    ax1.pie(dataset.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
                    ax1.axis('equal')
                    st.pyplot(fig1)

                with sesudah_smote:
                    st.header('sesudah SMOTE')
                    df = pd.DataFrame(Xsmote)
                    df.rename(columns={0:'term'}, inplace=True)
                    df['sentimen'] = Ysmote
                    #melihat banyak dataset
                    st.write('Jumlah dataset :',len(Ysmote))
                    # melihat jumlah kelas sentimen aetelah SMOTE
                    st.write("Jumlah kelas: ")
                    dJumlah_sentimen = df['sentimen'].value_counts()
                    pos, neg, net = st.columns(3)
                    with pos:
                        st.markdown("Positif")
                        st.markdown(f"<h1 style='text-align: center; color: green;'>{dJumlah_sentimen[1]}</h1>", unsafe_allow_html=True)
                    with net:
                        st.markdown("Netral")
                        st.markdown(f"<h1 style='text-align: center; color: orange;'>{dJumlah_sentimen[0]}</h1>", unsafe_allow_html=True)
                    with neg:
                        st.markdown("Negatif")
                        st.markdown(f"<h1 style='text-align: center; color: blue;'>{dJumlah_sentimen[-1]}</h1>", unsafe_allow_html=True)
                    
                    # menampilkan dalam bentuk plot diagram
                    labels = ['negatif' , 'neutral', 'positif']
                    fig2,ax2=plt.subplots()
                    plt.pie(df.groupby('sentimen')['sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
                    ax2.axis('equal')
                    st.pyplot(fig2)
                
                df = pd.DataFrame(Xsmote)
                df.rename(columns={0:'term'}, inplace=True)
                df['sentimen'] = Ysmote
                # mengembalikan kalimat asli dari tfidf
                feature_names = vectorizer.get_feature_names_out()

                kalimat_asli = []
                for index, row in df.iterrows():
                    vektor_ulasan = X_smote[index]
                    kata_kunci = [feature_names[i] for i in vektor_ulasan.indices]
                    kalimat_asli.append(' '.join(kata_kunci))

                # tambahkan kolom baru dengan kalimat asli ke dalam data frame
                df['kalimat_asli'] = kalimat_asli

                #mengambil data sintetik
                df_sintetik = df.iloc[1000:]

                dJum_sentimen = y.value_counts()
                st.header('Data sintetis')
                st.write('jumlah data sintetis :',len(df_sintetik))
                st.write('jumlah penambahan setiap kelas :')
                sinpos,sinneg,sinnet = st.columns(3)
                with sinpos:
                    st.markdown("Positif")
                    sel_pos=dJumlah_sentimen[1]-dJum_sentimen[1]
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{sel_pos}</h1>", unsafe_allow_html=True)
                with sinnet:
                    st.markdown("Netral")
                    sel_net=dJumlah_sentimen[0]-dJum_sentimen[0]
                    st.markdown(f"<h1 style='text-align: center; color: orange;'>{sel_net}</h1>", unsafe_allow_html=True)
                with sinneg:
                    st.markdown("Negatif")
                    sel_neg=dJumlah_sentimen[-1]-dJum_sentimen[-1]
                    st.markdown(f"<h1 style='text-align: center; color: blue;'>{sel_neg}</h1>", unsafe_allow_html=True)
                def sentimen_sintetis(dataset, Sentimen):
                    return df_sintetik[df_sintetik['sentimen'].isin(selected_sentiment)]
                def sentimen_smote(dataset, Sentimen):
                    return df[df['sentimen'].isin(sentiment)]
                optiondata = st.selectbox('pilih_data',('SMOTE_', 'SINTETIS_'))
                smote_,sintetis_ =st.tabs(['SMOTE_','SINTETIS_'])
                with smote_ :
                    # Panggil fungsi pencarian
                    st.write('menampilkan data sesudah',optiondata)
                    sentiment_map = {1: 'positif', 0: 'netral',-1:'negatif'}
                    sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
                    sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
                    filtered_data = sentimen_smote(df, sentiment)
                    st.dataframe(filtered_data)
                with sintetis_ :
                    st.write('menampilkan data ',optiondata)
                    sentiment_map = {1: 'positif', 0: 'netral'}
                    selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
                    selected_sentiment = [key for key, value in sentiment_map.items() if value in selected_sentiment]
                    filtered_data = sentimen_sintetis(df_sintetik, selected_sentiment)
                    st.dataframe(filtered_data)
            elif prepro == False :
                st.write('silahkan masukan dataset terlebih dahulu :)')

elif(selected == 'Algoritma Genetika') :
    st.title('Algoritma Genetika')
    st.text('jumlah populasi : 40')
    st.text('jumlah generasi : 20')
    st.text('jumlah crosover rate : 0.6')
    st.text('jumlah mutation rate :0.1')
    st.text('range parameter c :1-50')
    st.text('range parameter gamma :0.1-1')
    ga = pd.read_csv('data/GA_result.csv')
    st.dataframe(ga,use_container_width=True)

    best1 = ga['C_best'].iloc[-1]
    best2 = ga['Gamma_best'].iloc[-1]
    best3 = ga['Fitness_best'].iloc[-1]

    best_convergen = ga.loc[(ga['C_con'] == best1) & (ga['Gamma_con'] == best2) & (ga['Fitness_con'] == best3)]

    st.write(f"Kesimpulan dari tabel di atas yaitu : Algoritma Genetika terbaik Pada Generasi ke- { best_convergen['generasi'].values[0]} dengan nilai C :{round(best_convergen['C_con'].values[0],4)} ,nilai Gamma {round(best_convergen['Gamma_con'].values[0],4)},dan nilai Fitness {best_convergen['Fitness_con'].values[0]}")

    probcros = st.number_input('probabilitas crossover',format="%0.1f",value=0.6,step=0.1)
    probmutasi = st.number_input('probabilitas mutasi',format="%0.1f",value=0.2,step=0.1)
    populasi = st.number_input('banyak populasi',step=1,value=6)
    generasi = st.number_input('banyak generasi',step=1,value=3)

    if st.button('algoritma genetika') :
        with st.expander('data generasi') :
            x=X_smote
            y=Y_smote
            prob_crsvr = probcros 
            prob_mutation = probmutasi 
            population = populasi
            generations = generasi 

            kfold = 5

            x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,0,1,
                                0,1,1,1,0,0,1,0,1,1,1,0]) 


            pool_of_solutions = np.empty((0,len(x_y_string)))

            best_of_a_generation = np.empty((0,len(x_y_string)+1))

            for i in range(population):
                rd.shuffle(x_y_string)
                pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))

            gen = 1 
            gene=[]
            c_values = []
            gamma_values = []
            fitness_values = []
            cb_values = []
            gammab_values = []
            fitnessb_values = []
            for i in range(generations): 
                
                new_population = np.empty((0,len(x_y_string)))
                
                new_population_with_fitness_val = np.empty((0,len(x_y_string)+1))
                
                sorted_best = np.empty((0,len(x_y_string)+1))
                
                st.write()
                st.write()
                st.write("Generasi ke -", gen) 
                
                family = 1 
                
                for j in range(int(population/2)): 
                    
                    st.write()
                    st.write("populasi ke -", family) 
                    
                    parent_1 = svm_hp_opt.find_parents_ts(pool_of_solutions,x=x,y=y)[0]
                    parent_2 = svm_hp_opt.find_parents_ts(pool_of_solutions,x=x,y=y)[1]
                    
                    child_1 = svm_hp_opt.crossover(parent_1,parent_2,prob_crsvr=prob_crsvr)[0]
                    child_2 = svm_hp_opt.crossover(parent_1,parent_2,prob_crsvr=prob_crsvr)[1]
                    
                    mutated_child_1 = svm_hp_opt.mutation(child_1,child_2, prob_mutation=prob_mutation)[0]
                    mutated_child_2 = svm_hp_opt.mutation(child_1,child_2,prob_mutation=prob_mutation)[1]
                    
                    fitness_val_mutated_child_1 = svm_hp_opt.fitness(x=x,y=y,chromosome=mutated_child_1,kfold=kfold)[2]
                    fitness_val_mutated_child_2 = svm_hp_opt.fitness(x=x,y=y,chromosome=mutated_child_2,kfold=kfold)[2]
                    
                    

                    mutant_1_with_fitness_val = np.hstack((fitness_val_mutated_child_1,mutated_child_1)) 
                    
                    mutant_2_with_fitness_val = np.hstack((fitness_val_mutated_child_2,mutated_child_2)) 
                    
                    
                    new_population = np.vstack((new_population,
                                                mutated_child_1,
                                                mutated_child_2))
                    
                    
                    new_population_with_fitness_val = np.vstack((new_population_with_fitness_val,
                                                            mutant_1_with_fitness_val,
                                                            mutant_2_with_fitness_val))
                    
                    st.write(f"Parent 1:",str(parent_1))
                    st.write(f"Parent 2:",str(parent_2))
                    st.write(f"Child 1:",str(child_1))
                    st.write(f"Child 2:",str(child_2))
                    st.write(f"Mutated Child 1:",str(mutated_child_1))
                    st.write(f"Mutated Child 2:",str(mutated_child_2))
                    st.write(f"nilai fitness 1:",fitness_val_mutated_child_1)
                    st.write(f"nilai fitness 2:",fitness_val_mutated_child_2)

                    family = family+1
                pool_of_solutions = new_population
                
                sorted_best = np.array(sorted(new_population_with_fitness_val,
                                                        key=lambda x:x[0]))
                
                best_of_a_generation = np.vstack((best_of_a_generation,
                                                sorted_best[0]))
                
                
                sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                                    key=lambda x:x[0]))

                gen = gen+1 
                

                best_string_convergence = sorted_best[0]
                best_string_bestvalue = sorted_best_of_a_generation[0]
                final_solution_convergence = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_convergence[0:],kfold=kfold)
                final_solution_best = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_bestvalue[0:],kfold=kfold)
                
                gene.append(gen-1)

                c_values.append(final_solution_convergence[0])
                gamma_values.append(final_solution_convergence[1])
                fitness_values.append(final_solution_convergence[2])

                cb_values.append(final_solution_best[0])
                gammab_values.append(final_solution_best[1])
                fitnessb_values.append(final_solution_best[2])
                # create a dictionary to store the data
                results = {'generasi':gene,
                        'C_con': c_values,
                        'Gamma_con': gamma_values,
                        'Fitness_con': fitness_values,
                        'C_best': cb_values,
                        'Gamma_best': gammab_values,
                        'Fitness_best': fitnessb_values}
            sorted_last_population = np.array(sorted(new_population_with_fitness_val,key=lambda x:x[0]))

            sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,key=lambda x:x[0]))

            sorted_last_population[:,0] = (sorted_last_population[:,0]) # get accuracy instead of error
            sorted_best_of_a_generation[:,0] = (sorted_best_of_a_generation[:,0])
            best_string_convergence = sorted_last_population[0]

            best_string_overall = sorted_best_of_a_generation[0]

            # to decode the x and y chromosomes to their real values
            final_solution_convergence = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_convergence[1:],
                                                                    kfold=kfold)

            final_solution_overall = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_overall[1:],
                                                                kfold=kfold)
        # st.write(" C (Best):",str(round(final_solution_overall[0],4)) )# real value of x
        # st.write(" Gamma (Best):",str(round(final_solution_overall[1],4))) # real value of y
        # st.write(" fitness (Best):",str(round(final_solution_overall[2],4)) )# obj val of final chromosome
        # st.write()
        # st.write("------------------------------")
        result=pd.DataFrame(results)
        st.write(result)
        best1 = result['C_best'].iloc[-1]
        best2 = result['resultmma_best'].iloc[-1]
        best3 = result['Fitness_best'].iloc[-1]

        best_convergen = result.loc[(result['C_con'] == best1) & (result['resultmma_con'] == best2) & (result['Fitness_con'] == best3)]

        st.write(f"Kesimpulan dari tabel di atas yaitu : Algoritma Genetika terbaik Pada Generasi { best_convergen['generasi'].values[0]} denngan nilai C :{round(best_convergen['C_con'].values[0],4)} ,nilai Gamma {round(best_convergen['Gamma_con'].values[0],4)},dan nilai Fitness {best_convergen['Fitness_con'].values[0]}")

elif(selected == 'Pengujian') :
    st.header('pengujian model Support Vector Machine')
    with st.expander('pengujian parameter') :
        st.write('pengujian nilai kernel')
        pengujian_kernel = pd.read_csv('data/hasil_svm_kernel.csv')
        st.dataframe(pengujian_kernel,use_container_width=True)
        st.write('dari percobaan diatas kernel yg terbaik adalah kernel RBF')
        st.write('pengujian nilai C')
        pengujian_C = pd.read_csv('data/hasil_svm_C.csv')
        st.dataframe(pengujian_C,use_container_width=True)
        st.write('dari percobaan diatas di simpulkan bahwa nilai c yg efektif pada rentang 1-50')
        st.write('penjelasan nilai C')
        st.write('parameter C (juga dikenal sebagai parameter penalti) adalah faktor penting yang mempengaruhi kinerja dan perilaku model SVM. Parameter C mengendalikan trade-off antara penalti kesalahan klasifikasi dan lebar margin.Nilai C yang lebih kecil akan menghasilkan margin yang lebih lebar, dan menjadikan model tidak peka terhadap data dan kelasahan klasifikasi(UNDERFITTING). Sebaliknya, nilai C yang lebih besar akan menghasilkan margin yang lebih sempit, menjadikan model lebih peka terhadap data dan tingkat kesalahan klasifikasi.(OVERFITTING)')
        st.write('pengujian nilai Gamma')
        pengujian_Gamma = pd.read_csv('data/hasil_svm_gamma.csv')
        st.dataframe(pengujian_Gamma,use_container_width=True)
        st.write('dari percobaan diatas di simpulkan bahwa nilai gamma yg efektif pada rentang 0.1-0.99')
        st.write('penjelasan nilai gamma') 
        st.write('Parameter gamma dalam metode Support Vector Machine (SVM) mengontrol pengaruh dari satu sampel data terhadap pembentukan garis pemisah atau decision boundary.jika nilai gamma terlalu kecil makamodel akan terlalu sederhana dan menjadikan model menjadi tidak peka terhadap data dan ada kemungkinan overgeneralisasi, di mana model SVM dapat mengabaikan pola yang signifikan dalam data atau gagal memisahkan kelas yang berbeda secara efektif (UNDERFITTING). ,jika gamma terlalu besar maka model akan terlalu sensitif/detail terhadap data sehingga ada risiko overfitting, di mana model SVM dapat terlalu memperhatikan atau "menghafal" data pelatihan, dan performa pada data pengujian menjadi buruk.')
    st.title('analisis')
    option = st.selectbox('METODE',('SVM', 'GA-SVM'))
    document = st.text_input('masukan kalimat',value="Harusnya dikasih bintang 4 bilang terimakasih... Belum mau ngasih bintang 5. Soalnya jaringan di area saya sangat lemot. Terkadang dari 4G malah drop sampai ke E...")
    
    kcleansing = cleansing(document)
    kcasefolding = casefolding(kcleansing)
    ktokenizing = tokenizing(kcasefolding)
    kstemming = stemming(ktokenizing)
    knegasi= handle_negation(kstemming)
    kslangword = slangword(knegasi)
    kstopword = stopword(kslangword)
    kdatastr = str(kstopword)
    ktfidf =vectorizer.transform([kdatastr])

    if (option == 'SVM') :
        
        if st.button('predik') :
            st.write('Hasil pengujian dengan metode',option)
            # Making the SVM Classifer
            predictions = classifier.predict(ktfidf)

            st.write('hasil cleansing :',str(kcleansing))
            st.write('hasil casefolding :',str(kcasefolding))
            st.write('hasil tokenizing :',str(ktokenizing))
            st.write('hasil stemming :',str(kstemming))
            st.write('hasil negasi :',str(knegasi))
            st.write('hasil word normalization :',str(kslangword))
            st.write('hasil stopword :',str(kstopword))

            if not kstemming:
                st.write("Maaf mohon inputkan kalimat lagi :)")
            elif predictions == 1:
                st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen Positif")
            elif predictions == -1:
                st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen Negatif")
            elif predictions == 0:
                st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen netral")
        else:
            st.write('hasil akan tampil disini :)') 
    elif (option == 'GA-SVM') :
        if st.button('predik') :
            st.write('Hasil pengujian dengan metode',option)
            # Making the SVM Classifer
            predictions = svmga.predict(ktfidf)

            st.write('hasil cleansing :',str(kcleansing))
            st.write('hasil casefolding :',str(kcasefolding))
            st.write('hasil tokenizing :',str(ktokenizing))
            st.write('hasil stopword :',str(kstopword))
            st.write('hasil word normalization :',str(kslangword))
            st.write('hasil stemming :',str(kstemming))

            if not kstemming:
                st.write("Maaf mohon inputkan kalimat lagi :)")
            elif predictions == 1:
                st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen Positif")
            elif predictions == -1:
                st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen Negatif")
            elif predictions == 0:
                st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen netral")
        else:
            st.write('hasil akan tampil disini :)') 

elif(selected == 'Report') :
    st.title('evaluasi model')
    tab1,tab2 =st.tabs(['K-Fold Cross Validation', 'CONFUSION MATRIX'])
    norm_svm = svm.SVC()
    norm_svm.fit(X_train, Y_train)
    ga_svm = svm.SVC(kernel='rbf',C=12.7, gamma=0.3)
    ga_svm.fit(X_train, Y_train)
    snorm_svm = svm.SVC()
    snorm_svm.fit(sx_train, sy_train)
    sga_svm = svm.SVC(kernel='rbf',C=12.7, gamma=0.3)
    sga_svm.fit(sx_train, sy_train)

    with tab1 :
        st.header('akurasi')
        image = Image.open('data/akurasi_plot.png')
        st.image(image)
        st.write('dari plot yang ditampilkan di simpulkan bahwa nilai akurasi metode svm dengan optimasi parameter algoritma genetika lebih tinggi dibandingkan metode svm dengan nilai parameter default ')
        st.header('precision')
        image = Image.open('data/precision_plot.png')
        st.image(image)
        st.write('dari plot yang ditampilkan di simpulkan bahwa nilai presicion metode svm dengan optimasi parameter algoritma genetika lebih tinggi dibandingkan metode svm dengan nilai parameter default ')
        st.header('recall')
        image = Image.open('data/recall_plot.png')
        st.image(image)
        st.write('dari plot yang ditampilkan di simpulkan bahwa nilai recall metode svm dengan optimasi parameter algoritma genetika lebih tinggi dibandingkan metode svm dengan nilai parameter default ')
    with tab2 :
        st.title('sebelum SMOTE')
        st.header('Confusion matriks metode SVM')
        sy_pred = snorm_svm.predict(sx_test)
        f, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(confusion_matrix(sy_test, sy_pred), annot=True, fmt=".0f", ax=ax)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['negatif', 'netral','positif']); ax.yaxis.set_ticklabels(['negatif', 'netral','positif']);
        st.pyplot(f)
        st.header('Confusion matriks metode GA-SVM')
        sY_pred = sga_svm.predict(sx_test)
        f, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(confusion_matrix(sy_test, sY_pred), annot=True, fmt=".0f", ax=ax)
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['negatif', 'netral','positif']); ax.yaxis.set_ticklabels(['negatif', 'netral','positif']);
        st.pyplot(f)
        st.title('sesudah SMOTE')
        st.header('Confusion matriks metode SVM')
        y_pred = norm_svm.predict(X_test)
        f, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt=".0f", ax=ax)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['negatif', 'netral','positif']); ax.yaxis.set_ticklabels(['negatif', 'netral','positif']);
        st.pyplot(f)
        st.header('Confusion matriks metode GA-SVM')
        Y_pred = ga_svm.predict(X_test)
        f, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt=".0f", ax=ax)
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['negatif', 'netral','positif']); ax.yaxis.set_ticklabels(['negatif', 'netral','positif']);
        st.pyplot(f)




