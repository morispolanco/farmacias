import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

# Configuración inicial de la página
st.set_page_config(page_title="Inventory Insight - Farmacia Galeno", layout="wide")
st.title("Inventory Insight - Gestión Inteligente de Inventarios")

# Función para cargar y limpiar datos
def load_data(file):
    try:
        df = pd.read_csv(file)
        expected_columns = ['Fecha', 'Producto', 'Ventas', 'Stock', 'Fecha_Vencimiento']
        for col in expected_columns:
            if col not in df.columns:
                st.error(f"El archivo debe contener la columna: {col}")
                return None
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df['Fecha_Vencimiento'] = pd.to_datetime(df['Fecha_Vencimiento'], errors='coerce')
        df.dropna(subset=['Fecha', 'Ventas', 'Stock'], inplace=True)  # Eliminar filas con datos clave faltantes
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Función para pronosticar demanda con ARIMA
def forecast_demand(data, product, days=30):
    sales = data[data['Producto'] == product]['Ventas'].values
    if len(sales) < 10:  # Más datos para mejor precisión
        return None, "Necesitas al menos 10 días de datos históricos para un pronóstico fiable."
    try:
        model = ARIMA(sales, order=(1, 1, 1))  # Orden ajustable en versiones futuras
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        return forecast, None
    except Exception as e:
        return None, f"Error en el modelo de pronóstico: {e}"

# Función para generar recomendaciones de reabastecimiento
def suggest_restock(current_stock, predicted_demand, threshold, buffer=1.2):
    predicted_stock = current_stock - predicted_demand
    if predicted_stock < threshold:
        restock_amount = (predicted_demand * buffer) - current_stock  # Buffer para cubrir variaciones
        return max(restock_amount, 0)
    return 0

# Sidebar para configuraciones
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")

# Guía de uso
with st.sidebar.expander("¿Cómo usar esta aplicación?"):
    st.write("""
    **Inventory Insight** optimiza tu inventario:
    1. **Carga de Datos**: Sube un CSV con: Fecha, Producto, Ventas, Stock, Fecha_Vencimiento.
       - Ejemplo: `2025-01-01, Paracetamol, 10, 50, 2025-06-01`
    2. **Selecciona Producto**: Elige qué analizar.
    3. **Ajusta Parámetros**: Define días de pronóstico, umbral de stock y días de vencimiento.
    4. **Explora Resultados**:
       - Pronósticos de demanda con gráficos.
       - Estado del inventario con recomendaciones.
       - Alertas de vencimiento con acciones sugeridas.
    5. **Descarga Reporte**: Exporta los resultados.
    """)

# Opciones de personalización
forecast_days = st.sidebar.slider("Días de Pronóstico", 7, 90, 30)
stock_threshold = st.sidebar.number_input("Umbral de Stock Mínimo", min_value=0, value=10)
expiration_days = st.sidebar.slider("Días para Alerta de Vencimiento", 7, 90, 30)

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.sidebar.success("Datos cargados correctamente")
        products = data['Producto'].unique()
        selected_product = st.sidebar.selectbox("Selecciona un Producto", products)

        # Filtrar datos del producto
        product_data = data[data['Producto'] == selected_product]
        today = datetime.now()

        # Pronóstico de demanda
        st.subheader(f"Análisis para: {selected_product}")
        st.write("### Pronóstico de Demanda")
        with st.expander("¿Qué significa esto?"):
            st.write("""
            Este gráfico predice las ventas futuras usando un modelo estadístico (ARIMA):
            - **Azul**: Ventas pasadas.
            - **Rojo punteado**: Pronóstico.
            Úsalo para anticipar cuántas unidades necesitarás.
            """)

        forecast, error = forecast_demand(product_data, selected_product, forecast_days)
        if forecast is not None:
            forecast_dates = pd.date_range(start=product_data['Fecha'].max() + timedelta(days=1), 
                                          periods=forecast_days, freq='D')
            forecast_df = pd.DataFrame({'Fecha': forecast_dates, 'Pronóstico': forecast})

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(product_data['Fecha'], product_data['Ventas'], label='Ventas Históricas', color='blue')
            ax.plot(forecast_df['Fecha'], forecast_df['Pronóstico'], label='Pronóstico', color='red', linestyle='--')
            ax.fill_between(forecast_df['Fecha'], forecast * 0.9, forecast * 1.1, color='red', alpha=0.1, label='Margen de Error (±10%)')
            ax.legend()
            ax.set_title(f"Pronóstico de Demanda para {selected_product}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Ventas")
            st.pyplot(fig)
        else:
            st.warning(error)

        # Gestión de inventario
        st.write("### Gestión de Inventario")
        with st.expander("¿Qué significa esto?"):
            st.write("""
            Revisa tu inventario actual y futuro:
            - **Stock Actual**: Unidades disponibles.
            - **Demanda Pronosticada**: Ventas esperadas.
            - **Stock Esperado**: Stock tras la demanda.
            - **Recomendación**: Cantidad sugerida para reabastecer.
            """)

        current_stock = product_data['Stock'].iloc[-1]
        predicted_demand = forecast.sum() if forecast is not None else 0
        predicted_stock = current_stock - predicted_demand
        restock_amount = suggest_restock(current_stock, predicted_demand, stock_threshold)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stock Actual", int(current_stock))
        col2.metric("Demanda Pronosticada", int(predicted_demand))
        col3.metric("Stock Esperado", int(predicted_stock))
        col4.metric("Recomendación de Reabastecimiento", int(restock_amount))

        if predicted_stock < stock_threshold:
            st.warning(f"¡Alerta! El stock esperado ({int(predicted_stock)}) está por debajo del umbral ({stock_threshold}). Se recomienda reabastecer {int(restock_amount)} unidades.")

        # Control de vencimientos
        st.write("### Control de Vencimientos")
        with st.expander("¿Qué significa esto?"):
            st.write(f"""
            Identifica productos que vencerán en los próximos {expiration_days} días:
            - Prioriza ventas, aplica descuentos o considera donaciones.
            - Reduce pérdidas por productos expirados.
            """)

        expiration_data = product_data[['Fecha_Vencimiento', 'Stock']].dropna()
        expiration_threshold = today + timedelta(days=expiration_days)
        expiring_soon = expiration_data[expiration_data['Fecha_Vencimiento'] <= expiration_threshold]

        if not expiring_soon.empty:
            st.write(f"Productos próximos a vencer (en {expiration_days} días):")
            st.dataframe(expiring_soon.style.format({'Fecha_Vencimiento': '{:%Y-%m-%d}'}))
        else:
            st.success(f"No hay productos próximos a vencer en los próximos {expiration_days} días.")

        # Exportar reporte
        st.write("### Descargar Reporte")
        report_data = {
            'Producto': [selected_product],
            'Stock Actual': [current_stock],
            'Demanda Pronosticada': [predicted_demand],
            'Stock Esperado': [predicted_stock],
            'Reabastecimiento Sugerido': [restock_amount],
            'Productos por Vencer': [len(expiring_soon)]
        }
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Descargar Reporte CSV",
            data=csv,
            file_name=f"Reporte_{selected_product}_{today.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("Sube un archivo CSV con: 'Fecha', 'Producto', 'Ventas', 'Stock', 'Fecha_Vencimiento'.")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.write("Desarrollado por xAI para Farmacia Galeno - 2025")
