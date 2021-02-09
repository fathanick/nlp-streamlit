import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def createWordCloud():
    input_txt = st.text_area('Teks')

    if st.button('Buat Word Cloud') and input_txt != '':
        wcld = WordCloud().generate(input_txt)
        
        plt.figure(figsize = (8, 8), facecolor = None) 
        plt.imshow(wcld) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
  
        fig = plt.show() 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)


def abusiveDetection():
    input_txt = st.text_input('Tulis kalimat anda di sini')
    pkl_file = '../abusive-content-identification/svm_abusive_model.pkl'
    
    with open(pkl_file, 'rb') as file:
        loaded_model = pickle.load(file)

    if st.button('Cek') and input_txt != '':
        y_pred = loaded_model.predict([input_txt])
        if y_pred == 0:
            st.write('Konten kasar')
        elif y_pred == 1:
            st.write('Bukan konten kasar')


def sentimentDetection():
    input_txt = st.text_input('Tulis kalimat anda di sini')
    pkl_file = '../sentiment-analysis/sentiment_model.pkl'

    with open(pkl_file, 'rb') as file:
        loaded_model = pickle.load(file)

    if st.button('Cek Sentimen') and input_txt != '':
        y_pred = loaded_model.predict([input_txt])
        if y_pred == 0:
            st.write('Negatif')
        elif y_pred == 1:
            st.write('Positif')

def main():
    option = st.sidebar.selectbox(
        'Pilih Analisis Teks',
        ('Beranda','Deteksi Konten Kasar', 'Analisis Sentimen','Word Cloud')
    )

    if option == 'Beranda' or option == '':
        st.header('Welcome')
    elif option == 'Deteksi Konten Kasar':
        st.header('Deteksi Konten Kasar')
        abusiveDetection()
    elif option == 'Analisis Sentimen':
        st.header('Analisis Sentimen')
        sentimentDetection()
    elif option == 'Word Cloud':
        st.header('Word Cloud')
        createWordCloud()

if __name__ == '__main__':
    main()
   

