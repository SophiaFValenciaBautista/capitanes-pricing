import streamlit as st
import pandas as pd
import numpy as np

# 1. Configuración de la interfaz web
st.set_page_config(page_title="Capitanes CDMX Pricing App", layout="wide", page_icon="🏀")

st.title("🏀 Capitanes CDMX: Simulador de Pricing e Ingresos Arena CDMX")
st.markdown("""
Esta herramienta estratégica permite modelar la demanda y predecir los ingresos de taquilla 
utilizando el inventario real de asientos de la Arena CDMX. El modelo evalúa simultáneamente: 
**Precios, Categoría de Rival, Día de la Semana y Beneficios de Alimentos/Bebidas**.
""")

# 2. Parámetros del Modelo (Betas de Biogeme + Todas las Variables del Reto)
BETAS = {
    'B_VIP_MU': 0.1601,       
    'B_PREMIUM_MU': 0.1530,
    'B_ESTANDAR_MU': 0.3908,
    'B_PRECIO_MU': -0.2368,   
    'ASC_NINGUNO': -0.5200,
    
    # Coeficientes de Impacto de Escenario (Rival y Día)
    'B_RIVAL_TOP': 0.1145,      # Partido AAA (Gala)
    'B_RIVAL_MEDIO': 0.0450,    # Partido AA (Regular)
    'B_RIVAL_BAJO': -0.0600,    # Partido A (Baja Demanda)
    
    'B_FIN_DE_SEMANA': 0.0850,  # Bono por Viernes/Sábado/Domingo
    'B_ENTRE_SEMANA': -0.1200,  # Penalización por Lunes a Jueves
    
    # Coeficiente de Beneficio Agregado (Comida y Bebida)
    'B_COMIDA_BEBIDA': 0.1420   # Bono de utilidad por incluir alimentos/bebidas
}

CAPACIDAD_REAL = {
    'VIP': 1732,        
    'Premium': 3556,    
    'Estándar': 6373,   
    'Económica': 6376   
}
TOTAL_ASIENTOS = 18037

# 3. BARRA LATERAL - CONTROLES INTERACTIVOS COMPLETOS
st.sidebar.header("🕹️ Controles del Escenario")

# Control 1: Tipo de Partido
tipo_partido = st.sidebar.selectbox(
    "Categoría del Partido", 
    ["Partido de Gala (Rival Top - ej. Southbay Lakers)", 
     "Partido Regular (Rival Medio)", 
     "Partido de Baja Demanda (Rival del fondo de la tabla)"]
)

# Control 2: Día de la Semana
dia_semana = st.sidebar.radio(
    "Día del Encuentro",
    ["Fin de Semana (Vie - Dom)", "Entre Semana (Lun - Jue)"]
)

# Control 3: Beneficio de Alimentos y Bebidas (¡NUEVO!)
comida_bebida = st.sidebar.radio(
    "¿Incluye Alimentos y Bebidas ilimitados?",
    ["No (Solo boleto físico)", "Sí (Paquete con Alimentos y Bebidas incluidos)"]
)

st.sidebar.header("💵 Configuración de Precios (MXN)")
p_vip = st.sidebar.slider("Precio Zona VIP", 2499, 4999, 3800, step=100)
p_prem = st.sidebar.slider("Precio Zona Premium", 1499, 3499, 2400, step=50)
p_est = st.sidebar.slider("Precio Zona Estándar", 349, 799, 580, step=20)
p_econ = st.sidebar.slider("Precio Zona Económica", 49, 299, 180, step=10)

# Mapeo lógico de variables para el motor econométrico
if "Gala" in tipo_partido:
    rival_effect = BETAS['B_RIVAL_TOP']
elif "Regular" in tipo_partido:
    rival_effect = BETAS['B_RIVAL_MEDIO']
else:
    rival_effect = BETAS['B_RIVAL_BAJO']

dia_effect = BETAS['B_FIN_DE_SEMANA'] if "Fin" in dia_semana else BETAS['B_ENTRE_SEMANA']
comida_effect = BETAS['B_COMIDA_BEBIDA'] if "Sí" in comida_bebida else 0.0


# 4. MOTOR MATEMÁTICO LOGIT MULTIVARIANTE AVANZADO
def simular_demanda(p_vip, p_prem, p_est, p_econ, r_eff, d_eff, c_eff):
    # Ajustamos la escala del precio para que se alinee perfectamente con las constantes contextuales
    scale_precio = -0.2368 / 1000.0  # Beta de precio real por cada peso unitario
    
    # Factor de sensibilidad de mercado (para amplificar de forma realista los cambios de escenario)
    sensibilidad_mercado = 2.5
    
    # 1. Cálculo de Utilidades Estructurales (Precios escalados + Bonos directos de escenario)
    # Se añade el efecto de rival, día y comida de manera directa para que tengan peso real
    V_vip = (BETAS['B_VIP_MU'] + (scale_precio * p_vip) + r_eff + d_eff + c_eff) * sensibilidad_mercado
    V_prem = (BETAS['B_PREMIUM_MU'] + (scale_precio * p_prem) + r_eff + d_eff + c_eff) * sensibilidad_mercado
    V_est = (BETAS['B_ESTANDAR_MU'] + (scale_precio * p_est) + r_eff + d_eff + c_eff) * sensibilidad_mercado
    V_econ = (0.0 + (scale_precio * p_econ) + r_eff + d_eff + c_eff) * sensibilidad_mercado # Base
    
    # Constante de No Asistencia calibrada para reaccionar como un termómetro
    V_none = (BETAS['ASC_NINGUNO'] + 0.1) * sensibilidad_mercado
    
    # 2. Aplicación de la Fórmula Logit Multinominal
    exp_v = np.exp([V_vip, V_prem, V_est, V_econ, V_none])
    probs = exp_v / np.sum(exp_v)
    
    demanda_teorica = {
        'VIP': int(probs[0] * TOTAL_ASIENTOS),
        'Premium': int(probs[1] * TOTAL_ASIENTOS),
        'Estándar': int(probs[2] * TOTAL_ASIENTOS),
        'Económica': int(probs[3] * TOTAL_ASIENTOS)
    }
    
    # 3. Asignación física con tope estricto de aforo
    asist_vip = min(demanda_teorica['VIP'], CAPACIDAD_REAL['VIP'])
    asist_prem = min(demanda_teorica['Premium'], CAPACIDAD_REAL['Premium'])
    asist_est = min(demanda_teorica['Estándar'], CAPACIDAD_REAL['Estándar'])
    asist_econ = min(demanda_teorica['Económica'], CAPACIDAD_REAL['Económica'])
    
    asistencia_total = asist_vip + asist_prem + asist_est + asist_econ
    fans_no_asisten = TOTAL_ASIENTOS - asistencia_total
    
    cuotas_reales = {
        'VIP': (asist_vip / TOTAL_ASIENTOS) * 100,
        'Premium': (asist_prem / TOTAL_ASIENTOS) * 100,
        'Estándar': (asist_est / TOTAL_ASIENTOS) * 100,
        'Económica': (asist_econ / TOTAL_ASIENTOS) * 100,
        'No Asiste': (fans_no_asisten / TOTAL_ASIENTOS) * 100
    }
    
    ingreso_total = (asist_vip * p_vip) + (asist_prem * p_prem) + (asist_est * p_est) + (asist_econ * p_econ)
    
    return cuotas_reales, ingreso_total, asistencia_total, [asist_vip, asist_prem, asist_est, asist_econ], fans_no_asisten
# Ejecución del motor con las palancas dinámicas seleccionadas
cuotas, ingreso_m, asistencia_f, lista_asistencia, no_asisten_fans = simular_demanda(p_vip, p_prem, p_est, p_econ, rival_effect, dia_effect, comida_effect)


# 5. DESPLIEGUE EN PANELS RESPONSILES (COMPACTO PARA LAPTOPS)
col1, col2, col3 = st.columns(3)
col1.metric("💰 Ingreso Proyectado", f"${ingreso_m/1e6:.2f}M MXN")
col2.metric("👥 Asistencia Real", f"{asistencia_f:,} / {TOTAL_ASIENTOS:,} fans", f"{asistencia_f/TOTAL_ASIENTOS*100:.1f}% Ocupación")
col3.metric("📉 Fuga de Demanda (No Asiste / Sin Cupo)", f"{no_asisten_fans:,} fans")

st.markdown("---")

# Gráfica 1: Ocupación Física Agrupada (Compacta, sin scroll horizontal)
st.subheader("🏟️ Ocupación Física de Asientos en la Arena")
df_asientos = pd.DataFrame({
    'Asientos Vendidos': lista_asistencia,
    'Capacidad Máxima': [CAPACIDAD_REAL['VIP'], CAPACIDAD_REAL['Premium'], CAPACIDAD_REAL['Estándar'], CAPACIDAD_REAL['Económica']]
}, index=['VIP', 'Premium', 'Estándar', 'Económica']).reset_index().rename(columns={'index': 'Zona'})

df_melted = df_asientos.melt(id_vars='Zona', var_name='Métrica', value_name='Cantidad')

import altair as alt
chart = alt.Chart(df_melted).mark_bar().encode(
    x=alt.X('Métrica:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y('Cantidad:Q', title='Número de Asientos'),
    color=alt.Color('Métrica:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']), legend=alt.Legend(title="Indicador")), 
    column=alt.Column('Zona:N', title='Zonas del Estadio', header=alt.Header(labelOrient='bottom'))
).properties(width=130, height=280)

st.altair_chart(chart, use_container_width=False)

st.markdown("---")

# Gráfica 2: Market Share Coherente (100% responsiva)
st.subheader("📈 Participación de Elección Real en el Mercado (%)")
df_cuotas = pd.DataFrame({
    'Zona': ['VIP', 'Premium', 'Estándar', 'Económica', 'No Asiste'],
    'Porcentaje (%)': [cuotas['VIP'], cuotas['Premium'], cuotas['Estándar'], cuotas['Económica'], cuotas['No Asiste']]
})
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
