#Importamos todas las librerías que vamos a necesitar:

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import xgboost
from xgboost.sklearn import XGBClassifier, XGBRegressor
from streamlit_folium import folium_static
import folium


#Encabezado e introducción:
st.title("Data Science - Desafío Integrador")
st.header("Grupo 2 - Análisis servicio ecobicis año 2019")
st.image("Bicis.jpg", use_column_width=True)
st.write("--------")
st.subheader("Propuesta de valor para la prestadora del servicio:")
st.write("-Obtener información descriptiva de valor para la planificación del servicio.")
st.write("-Obtener información descriptiva potencialmente monetizable.")
st.write("-Desarrollar un modelo que estime la cantidad de viajes a realizar en un determinado día.")
st.subheader("Propuesta de valor para el usuario:")
st.write("-Desarrollar un modelo que estime la probabilidad encontrar una bici disponible en una estación.")
st.write("--------")
st.subheader("Consideraciones preliminares:")
st.write("-En 2019 se identificaron 412 estaciones y se obtuvo su información geográfica.")
st.write("-Se recopiló información del 2015 al 2019 y se identificaron 253.000 usuarios dados de alta.")
st.write("-Se obtuvo información sobre 6.000.000 de viajes realizados.")
st.write("-Había 4.000 bicicletas en el sistema aproximadamente.")
st.write("-En el promedio el 10% de las unidades estaba en mantenimiento.")
st.write("--------")



#Mapa de las estaciones de EcoBicis:
st.subheader("Mapa Bs.As. - Estaciones Ecobicis 2019")

#Dataset con las coordenadas:
estaciones = pd.read_csv("Estaciones.csv")

#Centramos un mapa en CABA:
caba = folium.Map(location=[-34.61, -58.4655], zoom_start=12.1)

#Agregamos las estaciones como marcadores:
for i in estaciones.index:
    tooltip = estaciones["nombre_estacion_origen"][i]
    lat = estaciones["lat_estacion_origen"][i]
    long = estaciones["long_estacion_origen"][i]
    folium.Marker([lat, long], pop_up=tooltip, tooltip=tooltip).add_to(caba)

# Lo cargamos en Streamlit
folium_static(caba)
st.write("--------")


#Análisis Descriptivo:
st.title("Análisis Descriptivo:")
st.image("Viajes_Diario.jpg", use_column_width=True)
st.write("--------")
st.image("Heatmap_Viajes.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Duracion.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Genero.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Edad.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Edad_Gen.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Hora.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Hora_Gen.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Temporada.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Laborable.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Lluvia.jpg", use_column_width=True)
st.write("--------")
st.image("Viajes_Temperatura.jpg", use_column_width=True)
st.write("--------")


    
#Modelos predictivos Machine Learning: 

#Abrimos los modelos:

    
model = XGBRegressor()
model.load_model("viajes.json")

    

model_bici = XGBClassifier()
model_bici.load_model("bicis-disponibles.json")  

    


st.title("Modelos Desarrollados:")    
modelo = st.selectbox("Seleccione Modelo:", ["", "Cantidad de viajes por día (Prestadora)", "Disponibilidad de bici por estación (Usuario)"])
    

if modelo == "":
    pass

elif modelo == "Cantidad de viajes por día (Prestadora)":     
    #Modelo predicción de viajes:
    st.header('Modelo Predicción de Viajes')
    st.write("Este modelo estima en función de ciertos parámetros la cantidad total de viajes a realizar en un día.")
    st.subheader('Seleccione Parámetros:')

    def user_input_parameters():
        temperatura = st.slider("Temperatura", -10, 50, 20)

        #Laborable
        laborable = st.selectbox("Dia Laborable", ['Si', 'No'])
        tipo_dia = 1
        if laborable == 'No':
            tipo_dia = 0

        #Lluvia
        lluvia = st.selectbox("Lluvia", ['Si', 'No'])
        llueve = 0
        if lluvia == 'Si':
            llueve = 1       


        #Temporada:
        estacion = st.selectbox("Temporada", ['otono', 'primavera', 'verano', 'invierno'])
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
    
    st.write("--------") 
    st.subheader("Modelo Utilizado : XGBRegressor")
    st.subheader('Parámetros Elegidos')
    st.write(df)

    if st.button('Run'):
        st.write('La cantidad de viajes estimadas para ese día es:')
        st.success(model.predict(df).astype(int))
        st.image("Feature_Importance_Viajes.jpg", use_column_width=True)
    

else:
    # Modelo Disponibilidad de bicis
    st.header('Modelo Disponibilidad de Bicis')
    st.write("Este modelo estima en función de ciertos parámetros la probabilidad de encontrar una bici disponible en la estación                             seleccionada.")
    st.subheader('Seleccione Parámetros:')

    def user_input_parameters_bicis():

        #temperatura
        temperatura_bici = st.slider("Temp", -10, 50, 20)

        #Laborable
        laborable_bici = st.selectbox("Laborable", ['Si', 'No'])
        tipo_dia_bici = 0
        if laborable_bici == 'No':
            tipo_dia_bici = 1

        #Lluvia
        lluvia_bici = st.selectbox("Lluvias", ['Si', 'No'])
        llueve_bici = 0
        if lluvia_bici == 'Si':
            llueve_bici = 1

        datos_dia_bici = [temperatura_bici, tipo_dia_bici, llueve_bici]


        #Hora:
        hora_bici_lista = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        hora_bici = st.selectbox("Hora", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
        i = hora_bici - 1
        hora_bici_lista[i] = 1 


        #Estacion:
        estacion_bici = st.selectbox('Estacion', ['Plaza del Angel Gris', 'Federico Lacroze', 'Plaza Bruno Giordano', 'Villa del Parque',                                            'Diagonal Norte'])
        estacion_bici_lista = []
        if estacion_bici == 'Plaza del Angel Gris':
            estacion_bici_lista = [1,0,0,0]
        elif estacion_bici == 'Federico Lacroze':
            estacion_bici_lista = [0,1,0,0]
        elif estacion_bici == 'Plaza Bruno Giordano':
            estacion_bici_lista = [0,0,1,0]
        elif estacion_bici == 'Villa del Parque':
            estacion_bici_lista = [0,0,0,1]
        else:
            estacion_bici_lista = [0,0,0,0]



        columnas_bici = ['Temperatura','Lluvia','Tipo_Nolaborable','Hora_01','Hora_02','Hora_03','Hora_04','Hora_05','Hora_06','Hora_07',
                    'Hora_08','Hora_09','Hora_10','Hora_11','Hora_12','Hora_13','Hora_14','Hora_15','Hora_16','Hora_17','Hora_18','Hora_19',
                    'Hora_20','Hora_21','Hora_22','Hora_23','id_estacion_148BAEcobici','id_estacion_175BAEcobici','id_estacion_349BAEcobici',
                    'id_estacion_94BAEcobici']
        features = datos_dia_bici + hora_bici_lista + estacion_bici_lista
        
        
        #Creamos un dataframe con los parametros elegidos:
        parametros_bici = pd.DataFrame(dict(zip(["Temperatura", "Lluvia", "Laborable", "Hora", "Estación"], [temperatura_bici, laborable_bici,                                                    lluvia_bici, hora_bici, estacion_bici])), index=[0])
        
        #Creamos el dataframe con las features para el modelo
        data_bici = dict(zip(columnas_bici, features))
        features_bici = pd.DataFrame(data_bici, index=[0])
        
        return parametros_bici, features_bici
    
    features_bici = user_input_parameters_bicis()

    st.subheader("Modelo : XGBRegressor")
    st.subheader('Parámetros Elegidos')
    st.write(features_bici[0])

    if st.button('Ejecutar'):
        st.write('Lo más probable es que la Estación seleccionada se encuentre:')
        if model_bici.predict(features_bici[1]) == 1:
            st.success("Con bicis disponibles")
        else:
            st.success("Sin bicis disponibles")
        st.image("Feature_Importance_Bicis.jpg", use_column_width=True)
    
    



