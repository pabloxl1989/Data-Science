#Importamos todas las librerías que vamos a necesitar:

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import xgboost

#Abrimos el archivo pickle
with open('xgbr.pkl', 'rb') as xgbr:
    model = pickle.load(xgbr)

    
    
    
def main():
    st.title('Modelo Predicción de Viajes')
    st.sidebar.header('Seleccione Parámetros:')
    
    def user_input_parameters():
        temperatura = st.sidebar.slider("Temperatura", -10, 50, 20)
        laborable = st.sidebar.select_slider("Laborable (0=No, 1=Si)", options=[0,1])
        lluvia = st.sidebar.select_slider("Lluvia (0=No, 1=Si)", options=[0,1])
        otono = st.sidebar.select_slider("Otono (0=No, 1=Si)", options=[0,1])
        primavera = st.sidebar.select_slider("Primavera (0=No, 1=Si)", options=[0,1])
        verano = st.sidebar.select_slider("Verano (0=No, 1=Si)", options=[0,1])
        data = {'temperatura': temperatura,
                'laborable': laborable,
                'lluvia': lluvia,
                'otono': otono,
                'primavera': primavera,
                'verano': verano
                }

        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()
    
    st.subheader("Modelo : XGBRegressor")
    st.subheader('Parámetros Elegidos')
    st.write(df)
    
    if st.button('Run'):
        st.write('La cantidad de viajes estimadas para ese día es:')
        st.success(model.predict(df))
if __name__ == '__main__':
    main()


