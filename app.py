import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración inicial de la página
st.set_page_config(page_title="Inventory Insight - Farmacia Galeno", layout="wide")
st.title("Inventory Insight - Gestión Inteligente de Inventarios")

# Función para cargar y limpiar datos desde CSV
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
    if len(sales) < 10:
        return None, "Necesitas al menos 10 días de datos históricos para un pronóstico fiable."
    
    try:
        model = ARIMA(sales, order=(1, 1, 1))  # Orden ajustable según necesidades
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        forecast_dates = pd.date_range(start=data['Fecha'].max() + timedelta(days=1), periods=days, freq='D')
        forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast})
        return forecast_df, None
    except Exception as e:
        return None, f"Error en el modelo de pronóstico: {e}"

# Función para generar recomendaciones de reabastecimiento
def suggest_restock(current_stock, predicted_demand, threshold, buffer=1.2):
    predicted_stock = current_stock - predicted_demand
    if predicted_stock < threshold:
        restock_amount = (predicted_demand * buffer) - current_stock
        return max(restock_amount, 0)
    return 0

# Función para enviar notificaciones por correo electrónico
def send_email(subject, body, to_email):
    from_email = "tu_correo@gmail.com"  # Cambia esto por tu correo
    password = "tu_contraseña"  # Cambia esto por tu contraseña

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        st.success("Correo enviado exitosamente.")
    except Exception as e:
        st.error(f"Error al enviar el correo: {e}")

# Datos simulados para información adicional del producto
product_info = {
    "Paracetamol": {"precio": "$5.99", "descripcion": "Analgésico común para el dolor."},
    "Ibuprofeno": {"precio": "$7.49", "descripcion": "Antiinflamatorio no esteroideo."},
    "Aspirina": {"precio": "$4.99", "descripcion": "Medicamento para dolores leves."},
}

# Función para obtener información simulada del producto
def fetch_product_info(product_name):
    info = product_info.get(product_name, {"precio": "N/A", "descripcion": "Información no disponible."})
    return info["precio"], info["descripcion"]

# Función para calcular la rotación de inventario
def calculate_inventory_turnover(data, product):
    total_sales = data[data['Producto'] == product]['Ventas'].sum()
    average_stock = data[data['Producto'] == product]['Stock'].mean()
    return total_sales / average_stock if average_stock > 0 else 0

# Función para obtener los productos más vendidos
def get_top_selling_products(data, top_n=5):
    top_products = data.groupby('Producto')['Ventas'].sum().nlargest(top_n).reset_index()
    return top_products

# Función para sugerir estrategias de compra basadas en costos
def suggest_purchase_strategy(current_stock, predicted_demand, threshold, buffer_factor, unit_cost, bulk_discount_threshold, bulk_discount_rate):
    restock_amount = suggest_restock(current_stock, predicted_demand, threshold, buffer_factor)
    total_cost = restock_amount * unit_cost
    
    if restock_amount >= bulk_discount_threshold:
        discounted_cost = total_cost * (1 - bulk_discount_rate)
        savings = total_cost - discounted_cost
        return restock_amount, discounted_cost, savings
    return restock_amount, total_cost, 0

# Sidebar para configuraciones
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")

forecast_days = st.sidebar.slider("Días de Pronóstico", 7, 90, 30)
stock_threshold = st.sidebar.number_input("Umbral de Stock Mínimo", min_value=0, value=10)
expiration_days = st.sidebar.slider("Días para Alerta de Vencimiento", 7, 90, 30)
buffer_factor = st.sidebar.number_input("Factor de Buffer (1.0-2.0)", min_value=1.0, max_value=2.0, value=1.2)

bulk_discount_threshold = st.sidebar.number_input("Umbral de Descuento por Volumen", min_value=0, value=100)
bulk_discount_rate = st.sidebar.number_input("Tasa de Descuento por Volumen (%)", min_value=0.0, max_value=1.0, value=0.1)

if uploaded_file:
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
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(product_data['Fecha'], product_data['Ventas'], label='Ventas Históricas', color='blue')
            ax.plot(forecast['ds'], forecast['yhat'], label='Pronóstico', color='red', linestyle='--')
            ax.fill_between(forecast['ds'], forecast['yhat'] * 0.9, forecast['yhat'] * 1.1, color='red', alpha=0.1, label='Margen de Error (±10%)')
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
        predicted_demand = forecast['yhat'].sum() if forecast is not None else 0
        predicted_stock = current_stock - predicted_demand
        restock_amount = suggest_restock(current_stock, predicted_demand, stock_threshold, buffer_factor)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stock Actual", int(current_stock))
        col2.metric("Demanda Pronosticada", int(predicted_demand))
        col3.metric("Stock Esperado", int(predicted_stock))
        col4.metric("Recomendación de Reabastecimiento", int(restock_amount))
        
        if predicted_stock < stock_threshold:
            st.warning(f"¡Alerta! El stock esperado ({int(predicted_stock)}) está por debajo del umbral ({stock_threshold}). Se recomienda reabastecer {int(restock_amount)} unidades.")
            send_email(
                subject=f"Alerta de Stock Bajo para {selected_product}",
                body=f"El stock esperado para {selected_product} es {int(predicted_stock)}, por debajo del umbral ({stock_threshold}). Se recomienda reabastecer {int(restock_amount)} unidades.",
                to_email="farmacia_galeno@example.com"  # Cambia esto por el correo de destino
            )

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

        # Información simulada del producto
        precio, descripcion = fetch_product_info(selected_product)
        st.write(f"### Información Adicional para {selected_product}")
        col1, col2 = st.columns(2)
        col1.metric("Precio Actual", precio)
        col2.write(f"Descripción: {descripcion}")

        # Panel de Control
        st.subheader("Panel de Control - Métricas Clave")

        # Rotación de Inventario
        inventory_turnover = calculate_inventory_turnover(data, selected_product)
        st.metric("Rotación de Inventario", round(inventory_turnover, 2))

        # Productos Más Vendidos
        top_products = get_top_selling_products(data)
        st.write("### Productos Más Vendidos")
        st.dataframe(top_products)

        # Stock Promedio
        average_stock = data.groupby('Producto')['Stock'].mean().reset_index()
        st.write("### Stock Promedio por Producto")
        st.bar_chart(average_stock.set_index('Producto'))

        # Estrategia de Compra
        unit_cost = float(precio.replace("$", "").replace(",", "")) if precio != "N/A" else 0
        restock_amount, total_cost, savings = suggest_purchase_strategy(
            current_stock, predicted_demand, stock_threshold, buffer_factor, unit_cost, bulk_discount_threshold, bulk_discount_rate
        )

        st.write("### Estrategia de Compra Recomendada")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cantidad a Reabastecer", int(restock_amount))
        col2.metric("Costo Total", f"${total_cost:.2f}")
        col3.metric("Ahorro por Descuento", f"${savings:.2f}")

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
