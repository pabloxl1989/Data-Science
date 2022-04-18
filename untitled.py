def main():
    
    #Modelo predicción de viajes:
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
        estacion = st.sidebar.selectbox("Temporada", ['otono', 'primavera', 'verano', 'invierno'])
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
        
    # Modelo Disponibilidad de bicis
    st.title('Modelo Disponibilidad de Bicis')
    st.header('Seleccione Parámetros:')
    
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
        estacion_bici = st.selectbox("Estacion", ["148BAEcobici", "175BAEcobici", "349BAEcobici", "94BAEcobici", "104BAEcobici"])
        estacion_bici_lista = []
        if estacion_bici == '148BAEcobici':
            estacion_bici_lista = [1,0,0,0]
        elif estacion_bici == '175BAEcobici':
            estacion_bici_lista = [0,1,0,0]
        elif estacion_bici == '349BAEcobici':
            estacion_bici_lista = [0,0,1,0]
        elif estacion_bici == '94BAEcobici':
            estacion_bici_lista = [0,0,0,1]
        else:
            estacion_bici_lista = [0,0,0,0]
        


        columnas_bici = ['Temperatura','Lluvia','Tipo_Nolaborable','Hora_01','Hora_02','Hora_03','Hora_04','Hora_05','Hora_06','Hora_07',
                    'Hora_08','Hora_09','Hora_10','Hora_11','Hora_12','Hora_13','Hora_14','Hora_15','Hora_16','Hora_17','Hora_18','Hora_19',
                    'Hora_20','Hora_21','Hora_22','Hora_23','id_estacion_148BAEcobici','id_estacion_175BAEcobici','id_estacion_349BAEcobici',
                    'id_estacion_94BAEcobici']
        parametros_bici = datos_dia_bici + hora_bici_lista + estacion_bici_lista
        data_bici = dict(zip(columnas_bici, parametros_bici))

        features_bici = pd.DataFrame(data_bici, index=[0])
        return features_bici
    
    df_bici = user_input_parameters_bicis()
    
    st.subheader("Modelo : XGBRegressor")
    st.subheader('Parámetros Elegidos')
    st.write(df_bici)
    
    if st.button('Ejecutar'):
        st.write('La probabilidad de encontrar bici para ese día en esa estación es:')
        st.success(model_bici.predict(df_bici))

if __name__ == '__main__':
    main()
