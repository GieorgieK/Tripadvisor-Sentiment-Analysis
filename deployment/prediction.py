import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Unduh sumber daya NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Muat model
model = load_model('best_model.keras')

# Definisikan stopwords dan lemmatizer
stopwords_nltk = list(set(stopwords.words('english')))
stopwords_add = ['hotel', 'room']
stopwords_all = stopwords_nltk + stopwords_add
lemmatizer = WordNetLemmatizer()

# Muat atau definisikan tokenizer
tokenizer = Tokenizer()
# Fit the tokenizer on your training data
# Contoh:
# tokenizer.fit_on_texts(training_texts)

# Definisikan fungsi pre-processing
def text_preprocessing(document):
    # Mengubah teks menjadi huruf kecil
    document = document.lower()
    # Memperbaiki kontraksi
    document = contractions.fix(document)
    # Menghapus tanda baca
    document = re.sub(f'[{re.escape(string.punctuation)}]', '', document)
    # Menghapus angka
    document = re.sub(r'\w*\d\w*', '', document)
    # Menghapus karakter non-ASCII
    document = re.sub('[^\x00-\x7f]', '', document)
    # Menghapus kata-kata pendek
    document = re.sub(r'\b\w{1,3}\b', ' ', document)
    # Menghapus spasi berlebih
    document = document.strip()
    # Tokenisasi
    tokens = word_tokenize(document)
    # Menghapus stopwords
    tokens = [word for word in tokens if word not in stopwords_all]
    # Lematisasi
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Menggabungkan token
    document = ' '.join(tokens)
    return document

def run():
    st.title("Analisis Sentimen Review")

    st.subheader("Prediksi")

    # Buat form input
    with st.form("Analisis Review"):
        review = st.text_area("Masukkan review:", "Masukkan review di sini", height=200)
        submitted = st.form_submit_button('Prediksi')

    df_inf = pd.DataFrame({'review': review}, index=[0])

    # Pre-proses review
    df_inf['review_processed'] = df_inf['review'].apply(lambda x: text_preprocessing(x))

    if submitted:
        # Pastikan review yang diproses tidak kosong
        if not df_inf['review_processed'][0]:
            st.write("Review yang dimasukkan terlalu pendek atau hanya berisi stopwords.")
            return

        # Buat input dengan bentuk yang diharapkan oleh TextVectorization
        X = np.array(df_inf['review_processed']).reshape(-1, 1)

        # Prediksi sentimen
        y_pred_inf_proba = model.predict(X)
        y_pred_inf = np.argmax(y_pred_inf_proba, axis=-1)

        # Tampilkan hasil prediksi
        if y_pred_inf == 0:
            st.write('Review ini negatif.')
        elif y_pred_inf == 1:
            st.write('Review ini netral.')
        else:
            st.write('Review ini positif.')

if __name__ == "__main__":
    run()