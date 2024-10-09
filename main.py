import streamlit as st
import numpy as np
import joblib

# Carregar o modelo treinado
model = joblib.load('modelo_random_forest.pkl')

st.title('Previsão do Efetivo de Rebanhos')

# Entradas do usuário
suino_total = st.number_input('Suíno - total')
equino = st.number_input('Equino')
bubalino = st.number_input('Bubalino')

# Prever com base nas entradas do usuário
if st.button('Prever'):
    features = np.array([[suino_total, equino, bubalino]])  # Adicione mais variáveis conforme necessário
    prediction = model.predict(features)
    st.write(f'Previsão de Bovinos: {prediction[0]}')