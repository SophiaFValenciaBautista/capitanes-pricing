import streamlit as st
import pandas as pd
import numpy as np

# 1. Configuración de la interfaz web
st.set_page_config(page_title="Capitanes CDMX Pricing App", layout="wide", page_icon="🏀")

st.title("🏀 Capitanes CDMX: Simulador de Pricing e Ingresos Arena CDMX")
st.markdown("""
Esta herramienta estratégica permite modelar la demanda y predecir los ingresos de taquilla 
utilizando el inventario real de asientos de la Arena CDMX y un modelo econométrico calibrado de elección discreta.
""")

# 2. Parámetros del Modelo (Betas de Biogeme e Inventario Real)
BETAS = {
    'B_VIP_MU': 0.1601,       
    'B_PREMIUM_MU': 0.1530,
    'B_ESTANDAR_MU': 0.3908,
    'B_PRECIO_MU': -0.2368,   
    'B_RIVAL_TOP': 0.1145,
    'ASC_NINGUNO': -0.5200    
}

CAPACIDAD_REAL = {
    'VIP': 1732,        
    'Premium': 3556,    
    'Estándar': 6373,   
    'Económica': 6376   
}
TOTAL_ASIENTOS = 18037

# 3. BARRA LATERAL - CONTROLES INTERACTIVOS
st.sidebar.header("🕹️ Controles del Escenario")
tipo_partido = st.sidebar.selectbox("Tipo de Partido", ["Partido Regular", "Rival Top (Southbay Lakers)"])

st.sidebar.header("💵 Configuración de Precios (MXN)")
p_vip = st.sidebar.slider("Precio Zona VIP", 2499, 4999, 3800, step=100)
p_prem = st.sidebar.slider("Precio Zona Premium", 1499, 3499, 2400, step=50)
p_est = st.sidebar.slider("Precio Zona Estándar", 349, 799, 580, step=20)
p_econ = st.sidebar.slider("Precio Zona Económica", 49, 299, 180, step=10)

rival_val = 1 if tipo_partido == "Rival Top (Southbay Lakers)" else 0

# 4. MOTOR MATEMÁTICO LOGIT RECALIBRADO (EL PUNTO MEDIO)
def simular_demanda(p_vip, p_prem, p_est, p_econ, rival_flag):
    # Parámetros de calibración fina para corregir la escala del mercado
    mu_escala = 4.5  # Modulador de varianza para activar la sensibilidad de VIP/Premium
    
    # Cálculo de utilidades base (Precios en escala de miles de acuerdo al modelo)
    V_vip = BETAS['B_VIP_MU'] + (BETAS['B_PRECIO_MU'] * (p_vip / 1000.0)) + (BETAS['B_RIVAL_TOP'] * rival_flag)
    V_prem = BETAS['B_PREMIUM_MU'] + (BETAS['B_PRECIO_MU'] * (p_prem / 1000.0)) + (BETAS['B_RIVAL_TOP'] * rival_flag)
    V_est = BETAS['B_ESTANDAR_MU'] + (BETAS['B_PRECIO_MU'] * (p_est / 1000.0)) + (BETAS['B_RIVAL_TOP'] * rival_flag)
    V_econ = 0.0 + (BETAS['B_PRECIO_MU'] * (p_econ / 1000.0)) + (BETAS['B_RIVAL_TOP'] * rival_flag) # Zona base
    
    # Umbral de no asistencia ajustado para que reaccione dinámicamente
    V_none = BETAS['ASC_NINGUNO'] - 1.2 
    
    # Aplicación del multiplicador de escala a las utilidades
    exp_v = np.exp(np.array([V_vip, V_prem, V_est, V_econ, V_none]) * mu_escala)
    probs = exp_v / np.sum(exp_v)
    
    # Distribución inicial de la demanda teórica basada en las probabilidades
    demanda_teorica = {
        'VIP': int(probs[0] * TOTAL_ASIENTOS),
        'Premium': int(probs[1] * TOTAL_ASIENTOS),
        'Estándar': int(probs[2] * TOTAL_ASIENTOS),
        'Económica': int(probs[3] * TOTAL_ASIENTOS),
        'Ninguno': int(probs[4] * TOTAL_ASIENTOS)
    }
    
    # APLICACIÓN ESTRICTA DEL TOPE FÍSICO
    asist_vip = min(demanda_teorica['VIP'], CAPACIDAD_REAL['VIP'])
    asist_prem = min(demanda_teorica['Premium'], CAPACIDAD_REAL['Premium'])
    asist_est = min(demanda_teorica['Estándar'], CAPACIDAD_REAL['Estándar'])
    asist_econ = min(demanda_teorica['Económica'], CAPACIDAD_REAL['Económica'])
    
    # La asistencia total real es la suma de los asientos que sí se pudieron vender
    asistencia_total = asist_vip + asist_prem + asist_est + asist_econ
    
    # Cualquier persona que se quedó sin lugar por tope físico se acumula lógicamente en 'No Asiste'
    fans_no_asisten = TOTAL_ASIENTOS - asistencia_total
    
    # Recalcular las cuotas reales del gráfico de pastel/barras de elección
    cuotas_reales = {
        'VIP': (asist_vip / TOTAL_ASIENTOS) * 100,
        'Premium': (asist_prem / TOTAL_ASIENTOS) * 100,
        'Estándar': (asist_est / TOTAL_ASIENTOS) * 100,
        'Económica': (asist_econ / TOTAL_ASIENTOS) * 100,
        'No Asiste': (fans_no_asisten / TOTAL_ASIENTOS) * 100
    }
    
    # Cálculo preciso de los ingresos reales en caja
    ingreso_total = (asist_vip * p_vip) + (asist_prem * p_prem) + (asist_est * p_est) + (asist_econ * p_econ)
    
    return cuotas_reales, ingreso_total, asistencia_total, [asist_vip, asist_prem, asist_est, asist_econ], fans_no_asisten

# Ejecución del motor antes del renderizado visual
cuotas, ingreso_m, asistencia_f, lista_asistencia, no_asisten_fans = simular_demanda(p_vip, p_prem, p_est, p_econ, rival_val)

# 5. DESPLIEGUE EN PANELS (FRONTEND VISUAL CORREGIDO)
col1, col2, col3 = st.columns(3)
col1.metric("💰 Ingreso Proyectado", f"${ingreso_m/1e6:.2f}M MXN")
col2.metric("👥 Asistencia Real (Boletos Vendidos)", f"{asistencia_f:,} / {TOTAL_ASIENTOS:,} fans", f"{asistencia_f/TOTAL_ASIENTOS*100:.1f}% Ocupación")
col3.metric("📉 Fuga de Demanda (No Asiste / Sin Cupo)", f"{no_asisten_fans:,} fans")

st.markdown("---")

# Gráfica 1: Ocupación Física (Ajustada para laptops)
st.subheader("🏟️ Ocupación Física de Asientos en la Arena")

# Reestructuración del DataFrame
df_asientos = pd.DataFrame({
    'Asientos Vendidos': lista_asistencia,
    'Capacidad Máxima': [CAPACIDAD_REAL['VIP'], CAPACIDAD_REAL['Premium'], CAPACIDAD_REAL['Estándar'], CAPACIDAD_REAL['Económica']]
}, index=['VIP', 'Premium', 'Estándar', 'Económica']).reset_index().rename(columns={'index': 'Zona'})

df_melted = df_asientos.melt(id_vars='Zona', var_name='Métrica', value_name='Cantidad')

# CONFIGURACIÓN ALTAIR COMPACTA PARA EVITAR SCROLL HORIZONTAL
import altair as alt
chart = alt.Chart(df_melted).mark_bar().encode(
    x=alt.X('Métrica:N', title=None, axis=alt.Axis(labels=False, ticks=False)), # Oculta texto interno innecesario para ahorrar espacio
    y=alt.Y('Cantidad:Q', title='Número de Asientos'),
    color=alt.Color('Métrica:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']), legend=alt.Legend(title="Indicador")), 
    column=alt.Column('Zona:N', title='Zonas del Estadio', header=alt.Header(labelOrient='bottom')) # Pone el nombre de la zona abajo, estilo pestaña
).properties(
    width=130,  # Ancho fijo y compacto por cada zona para que las 4 sumadas quepan en cualquier pantalla
    height=280
)

st.altair_chart(chart, use_container_width=False) # Forzamos el límite compacto diseñado arriba

st.markdown("---")

# Gráfica 2: Market Share (Ajustada al 100% del ancho de forma nativa)
st.subheader("📈 Participación de Elección Real en el Mercado (%)")
df_cuotas = pd.DataFrame({
    'Zona': ['VIP', 'Premium', 'Estándar', 'Económica', 'No Asiste'],
    'Porcentaje (%)': [cuotas['VIP'], cuotas['Premium'], cuotas['Estándar'], cuotas['Económica'], cuotas['No Asiste']]
})

# st.bar_chart nativo con set_index se adapta automáticamente al monitor sin desbordarse nunca
st.bar_chart(df_cuotas.set_index('Zona'), use_container_width=True)

st.markdown("---")

# Diagnóstico de Alerta Comercial
st.subheader("💡 Diagnóstico Estratégico Comercial")
porc_ocupacion = (asistencia_f / TOTAL_ASIENTOS) * 100
if porc_ocupacion > 85:
    st.success("🎯 ESCENARIO DE ALTO RENDIMIENTO: El mix de precios genera volumen óptimo de taquilla y alta asistencia.")
elif porc_ocupacion > 50:
    st.info("⚡ ESCENARIO EN EQUILIBRIO: Precios moderados. La asistencia es estable y permite una operación logística cómoda.")
else:
    st.error("🚨 ALERTA DE PRECIOS ALTOS: La demanda se está contrayendo. Se sugiere bajar tarifas para evitar un estadio vacío.")