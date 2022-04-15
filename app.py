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

        #Laborable
        laborable = st.sidebar.selectbox("Dia Laborable", ['Si', 'No'])
        tipo_dia = 1
        if laborable == 'No':
            tipo_dia = 0

        #Lluvia
        lluvia = st.sidebar.selectbox("Lluvia", ['Si', 'No'])
        llueve = 0
        if lluvia == 'Si':
            llueve = 1       
        
        
        #Temporada:
        estacion = st.sidebar.selectbox("Estacion", ['otono', 'primavera', 'verano', 'invierno'])
        temporada = []
        if estacion == 'otono':
            temporada = [1,0,0]
        elif estacion == 'primavera':
            temporada = [0,1,0]
        elif estacion == 'verano':
            temporada = [0,0,1]
        else:
            temporada = [0,0,0]
        


        data = {'temperatura': temperatura,
                'laborable': tipo_dia,
                'lluvia': llueve,
                'otono': temporada[0],
                'primavera': temporada[1],
                'verano': temporada[2]
                }

        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()
    
    st.subheader("Modelo : XGBRegressor")
    st.subheader('Parámetros Elegidos')
    st.write(df)
    
    if st.button('Run'):
        st.write('La cantidad de viajes estimadas para ese día es:')
        st.success(model.predict(df).astype(int))

if __name__ == '__main__':
    main()


