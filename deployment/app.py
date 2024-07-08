import eda
import prediction
import streamlit as st


page = st.sidebar.selectbox('Pilih Halaman: ', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.main()
else:
    prediction.run()