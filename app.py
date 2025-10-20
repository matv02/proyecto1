import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Mi Primera App")

# Subir archivo CSV
uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Leer el archivo CSV
    try:
        df = pd.read_csv(uploaded_file, header=0)
    except:
        df = pd.read_excel(uploaded_file, sheet_name="YEAR2")
        suma_un = df["UNETA"].sum()
        df_mes_actual = df[df["PERIODO"] == 12]


    # Mostrar la suma
    st.write(f"Suma de UNETA:{suma_un}")

    # Gráfico de líneas: PERIODO vs UNETA
    st.write("Gráfico de UNETA por PERIODO (mes actual):")
    st.line_chart(df.set_index("PERIODO")["UNETA"])

    # Mostrar el dataframe
    st.write("Dataframe de mes actual:")
    st.dataframe(df_mes_actual)

    st.write("Dataframe de resultados:")
    st.dataframe(df)
    
   