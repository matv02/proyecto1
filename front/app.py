import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import dukascopy_python
from dukascopy_python import (
    INTERVAL_HOUR_1,
    INTERVAL_DAY_1,
    OFFER_SIDE_BID,
    instruments
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Configuración inicial ---
st.set_page_config(page_title="📊 Análisis Técnico, Regresión y Pronóstico Dukascopy", layout="wide")
st.title("📈 Análisis Técnico + Regresión + Pronóstico con Datos de Dukascopy")

# --- Selección del activo ---
instrument_name = st.selectbox(
    "Selecciona un instrumento financiero:",
    [
        "INSTRUMENT_FX_MAJORS_EUR_USD",
        "INSTRUMENT_FX_MAJORS_GBP_USD",
        "INSTRUMENT_FX_MAJORS_USD_JPY",
        "INSTRUMENT_FX_MAJORS_AUD_USD"
    ],
    index=0
)
instrument = getattr(instruments, instrument_name)

# --- Selección de fechas e intervalo ---
start_date = st.date_input("Fecha inicial:", datetime(2025, 1, 1))
end_date = st.date_input("Fecha final:", datetime(2025, 2, 1))
interval_option = st.selectbox("Intervalo:", ["1 hora", "1 día"], index=0)
interval = INTERVAL_HOUR_1 if interval_option == "1 hora" else INTERVAL_DAY_1

# --- Botón principal ---
if st.button("📥 Obtener y analizar datos"):
    if start_date > end_date:
        st.error("❌ La fecha inicial no puede ser posterior a la final.")
    else:
        st.info("Cargando datos desde Dukascopy, por favor espera...")

        try:
            # --- Descarga de datos ---
            df = dukascopy_python.fetch(
                instrument=instrument,
                interval=interval,
                offer_side=OFFER_SIDE_BID,
                start=datetime.combine(start_date, datetime.min.time()),
                end=datetime.combine(end_date, datetime.min.time()),
            )

            df = df.reset_index().rename(columns={"timestamp": "Fecha"})
            df["Fecha"] = pd.to_datetime(df["Fecha"])

            # --- Indicadores técnicos ---
            if "close" in df.columns:
                df["SMA_20"] = df["close"].rolling(window=20).mean()
                df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df["RSI"] = 100 - (100 / (1 + rs))

                df["BB_Media"] = df["SMA_20"]
                df["BB_Sup"] = df["SMA_20"] + (df["close"].rolling(window=20).std() * 2)
                df["BB_Inf"] = df["SMA_20"] - (df["close"].rolling(window=20).std() * 2)

                ema12 = df["close"].ewm(span=12, adjust=False).mean()
                ema26 = df["close"].ewm(span=26, adjust=False).mean()
                df["MACD"] = ema12 - ema26
                df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

            st.session_state["df"] = df
            st.success(f"✅ Datos obtenidos correctamente ({len(df)} registros).")

            # --- Vista previa ---
            st.subheader("📋 Vista previa de los datos e indicadores")
            st.dataframe(df.tail(10))

            # --- Gráficos técnicos ---
            if "close" in df.columns:
                st.subheader("📈 Precio de cierre y medias móviles")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df["Fecha"], df["close"], label="Cierre", color="blue")
                ax.plot(df["Fecha"], df["SMA_20"], label="SMA 20", color="orange")
                ax.plot(df["Fecha"], df["EMA_20"], label="EMA 20", color="green")
                ax.legend()
                st.pyplot(fig)

                st.subheader("📉 RSI (Índice de Fuerza Relativa)")
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(df["Fecha"], df["RSI"], color="purple")
                ax.axhline(70, color="red", linestyle="--")
                ax.axhline(30, color="green", linestyle="--")
                st.pyplot(fig)

                st.subheader("📊 MACD (Tendencia y Momentum)")
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(df["Fecha"], df["MACD"], label="MACD", color="blue")
                ax.plot(df["Fecha"], df["Signal"], label="Señal", color="orange")
                ax.legend()
                st.pyplot(fig)

                st.subheader("🎯 Bandas de Bollinger")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df["Fecha"], df["close"], label="Cierre", color="blue")
                ax.plot(df["Fecha"], df["BB_Sup"], label="Banda Superior", color="red", linestyle="--")
                ax.plot(df["Fecha"], df["BB_Inf"], label="Banda Inferior", color="green", linestyle="--")
                ax.fill_between(df["Fecha"], df["BB_Inf"], df["BB_Sup"], color="gray", alpha=0.1)
                ax.legend()
                st.pyplot(fig)

            # --- Análisis de regresión ---
            st.subheader("📈 Análisis de Regresión Lineal Múltiple")

            features = ["SMA_20", "EMA_20", "RSI", "MACD", "Signal"]
            df_reg = df.dropna(subset=features + ["close"])

            X = df_reg[features]
            y = df_reg["close"]

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            st.write("**📊 Resultados del modelo:**")
            st.write(f"- R²: {r2:.4f}")
            st.write(f"- MAE: {mae:.6f}")
            st.write(f"- RMSE: {rmse:.6f}")

            coef_df = pd.DataFrame({
                "Variable": features,
                "Coeficiente": model.coef_
            })
            st.write("**🔢 Coeficientes del modelo:**")
            st.dataframe(coef_df)

            # --- Gráfico real vs predicho ---
            st.subheader("📉 Precio real vs predicho")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_reg["Fecha"], y, label="Real", color="blue")
            ax.plot(df_reg["Fecha"], y_pred, label="Predicho", color="red", alpha=0.7)
            ax.legend()
            st.pyplot(fig)

            # --- Pronóstico 8 periodos ---
            st.subheader("🔮 Pronóstico para los próximos 8 periodos")

            last_row = df_reg.iloc[-1]
            if interval == INTERVAL_HOUR_1:
                step = timedelta(hours=1)
            else:
                step = timedelta(days=1)

            future_dates = [last_row["Fecha"] + (i+1)*step for i in range(8)]
            last_features = last_row[features].values.reshape(1, -1)
            future_predictions = [model.predict(last_features)[0] for _ in range(8)]

            forecast_df = pd.DataFrame({
                "Fecha": future_dates,
                "Pronóstico_close": future_predictions
            })

            st.dataframe(forecast_df)

            # --- Gráfico combinado ---
            st.subheader("📊 Precio histórico + pronóstico")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_reg["Fecha"], df_reg["close"], label="Histórico", color="blue")
            ax.plot(forecast_df["Fecha"], forecast_df["Pronóstico_close"], label="Pronóstico", color="orange", marker="o")
            ax.legend()
            st.pyplot(fig)

            st.info("El pronóstico asume estabilidad temporal de los indicadores técnicos recientes.")

        except Exception as e:
            st.error(f"❌ Error al obtener los datos: {e}")