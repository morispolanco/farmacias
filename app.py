import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuración inicial de la página
st.set_page_config(page_title="Inventory Insight - Farmacia Galeno", layout="wide")
st.title("Inventory Insight - Gestión de Inventarios para Farmacia Galeno")

# Función para cargar y limpiar datos
def load_data(file):
    df = pd.read_csv(file)
    # Asegurarse de que las columnas esperadas estén presentes
    expected_columns = ['Fecha', 'Producto', 'Ventas', 'Stock', 'Fecha_Vencimiento']
    for col in expected_columns:
        if col not in df.columns:
            st.error(f"El archivo debe contener la columna: {col}")
            return None
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Fecha_Vencimiento'] = pd.to_datetime(df['Fecha_Vencimiento'])
    return df

# Función para pronosticar demanda con ARIMA
def forecast_demand(data, product, days=30):
    sales = data[data['Producto'] == product]['Ventas'].values
    if len(sales) < 5:  # Mínimo de datos para ARIMA
        return None
    model = ARIMA(sales, order=(1, 1, 1))  # Orden simple para demostración
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

# Sidebar para cargar archivo
st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.sidebar.success("Datos cargados correctamente")

        # Selección de producto
        products = data['Producto'].unique()
        selected_product = st.sidebar.selectbox("Selecciona un Producto", products)

        # Filtros y parámetros
        forecast_days = st.sidebar.slider("Días de Pronóstico", 7, 90, 30)
        stock_threshold = st.sidebar.number_input("Umbral de Stock Mínimo", min_value=0, value=10)

        # Sección principal
        st.subheader(f"Análisis para: {selected_product}")

        # Filtrar datos del producto seleccionado
        product_data = data[data['Producto'] == selected_product]

        # Pronóstico de demanda
        forecast = forecast_demand(product_data, selected_product, forecast_days)
        if forecast is not None:
            st.write("### Pronóstico de Demanda")
            forecast_dates = pd.date_range(start=product_data['Fecha'].max() + timedelta(days=1), 
                                          periods=forecast_days, freq='D')
            forecast_df = pd.DataFrame({'Fecha': forecast_dates, 'Pronóstico': forecast})

            # Gráfico de ventas históricas y pronóstico
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(product_data['Fecha'], product_data['Ventas'], label='Ventas Históricas', color='blue')
            ax.plot(forecast_df['Fecha'], forecast_df['Pronóstico'], label='Pronóstico', color='red', linestyle='--')
            ax.legend()
            ax.set_title(f"Pronóstico de Demanda para {selected_product}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Ventas")
            st.pyplot(fig)

        # Gestión de inventario
        current_stock = product_data['Stock'].iloc[-1]
        predicted_demand = forecast.sum() if forecast is not None else 0
        predicted_stock = current_stock - predicted_demand

        col1, col2, col3 = st.columns(3)
        col1.metric("Stock Actual", int(current_stock))
        col2.metric("Demanda Pronosticada", int(predicted_demand))
        col3.metric("Stock Esperado", int(predicted_stock))

        if predicted_stock < stock_threshold:
            st.warning(f"¡Alerta! El stock esperado ({int(predicted_stock)}) está por debajo del umbral ({stock_threshold}).")

        # Gestión de fechas de vencimiento
        st.write("### Control de Vencimientos")
        expiration_data = product_data[['Fecha_Vencimiento', 'Stock']].dropna()
        today = datetime.now()
        expiration_threshold = today + timedelta(days=30)

        expiring_soon = expiration_data[expiration_data['Fecha_Vencimiento'] <= expiration_threshold]
        if not expiring_soon.empty:
            st.write("Productos próximos a vencer (en 30 días):")
            st.dataframe(expiring_soon)
        else:
            st.success("No hay productos próximos a vencer en los próximos 30 días.")

else:
    st.info("Por favor, sube un archivo CSV con los datos de ventas y stock. Formato esperado: 'Fecha', 'Producto', 'Ventas', 'Stock', 'Fecha_Vencimiento'.")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.write("Desarrollado por xAI para Farmacia Galeno - 2025")
