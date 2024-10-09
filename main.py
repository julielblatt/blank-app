import streamlit as st
import numpy as np
import joblib

# Carregar o modelo treinado e o escalador
model = joblib.load('modelo_equinos_random_forest.pkl')
scaler = joblib.load('scaler_equinos.pkl')

st.title('Previsão do Número de Equinos')

# Entradas do usuário - Somente 3 colunas
suino_total = st.number_input('Suíno - total')
galinaceos_galinhas = st.number_input('Galináceos - galinhas')
bovino = st.number_input('Bovino')

# Organizar as entradas em um array
features = np.array([[suino_total, galinaceos_galinhas, bovino]])

# Aplicar a normalização nas entradas (usando o mesmo escalador do treinamento)
features_scaled = scaler.transform(features)

# Fazer a previsão
if st.button('Prever'):
    prediction = model.predict(features_scaled)
    st.write(f'Previsão de Equinos: {prediction[0]}')