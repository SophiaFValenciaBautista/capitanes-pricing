import streamlit as st
import pandas as pd
import numpy as np
import itertools
import altair as alt

# =========================================================
# SIMULADOR DE PRICING - CAPITANES CDMX
# Modelo: Choice-Based Conjoint + Logit (medias poblacionales)
# estimado en Biogeme. TODOS los betas provienen del modelo real
# HB_Capitanes_Real_4Zonas. No hay parametros inventados.
# =========================================================

st.set_page_config(page_title="Capitanes CDMX Pricing", layout="wide", page_icon="🏀")

st.title("🏀 Capitanes CDMX — Simulador de Pricing")
st.markdown("""
Herramienta de simulación de demanda e ingresos de taquilla para la Arena CDMX.
Basada en un modelo **Choice-Based Conjoint estimado en Biogeme** (medias poblacionales).
Todos los coeficientes provienen del modelo real; los precios simulados se mantienen
dentro del rango evaluado en la encuesta.
""")

# ---------------------------------------------------------
# BETAS REALES estimados en Biogeme (modelo HB_Capitanes_Real_4Zonas)
# Escala: el precio entra dividido entre 1000 (igual que en la estimacion).
# La zona Economica es la base (utilidad de zona = 0).
# Solo se usan las MEDIAS (MU): los SIGMA no convergieron con 1000 draws,
# por lo que la heterogeneidad individual NO se modela (queda como trabajo futuro).
# ---------------------------------------------------------
BETAS = {
    'B_VIP':      0.3655631700557071,
    'B_PREMIUM':  0.3494199836399673,
    'B_ESTANDAR': 0.8921894078121936,
    'B_PRECIO':  -0.5406648384012817,   # por cada $1,000 MXN
    'B_RIVAL_TOP':0.26153827558027254,
    'ASC_NINGUNO':-1.0402528716651978,
}

# Ajuste de modelo (para reportar honestamente)
MODELO_INFO = {
    'accuracy': 45.61,
    'n_obs': 1116,
    'rho2_nota': "Accuracy holdout 45.6% (azar con 4 opciones = 25%)",
}

# ---------------------------------------------------------
# Inventario real de la Arena CDMX (18,037 asientos)
# Mapeo de las 7 zonas comerciales a las 4 zonas del modelo.
# ---------------------------------------------------------
CAPACIDAD = {'VIP': 1732, 'Premium': 3556, 'Estándar': 6373, 'Económica': 6376}
TOTAL_ASIENTOS = sum(CAPACIDAD.values())  # 18,037

# Rangos de precio EVALUADOS EN LA ENCUESTA (zona segura, no extrapolar)
RANGOS = {
    'VIP':       (2499, 4999),
    'Premium':   (1499, 3499),
    'Estándar':  (349, 799),
    'Económica': (49, 299),
}

# =========================================================
# MOTOR: Logit multinomial con los betas reales. Sin factores de escala.
# =========================================================
def utilidades(p_vip, p_prem, p_est, p_econ, rival_top):
    bp = BETAS['B_PRECIO']
    r = BETAS['B_RIVAL_TOP'] * rival_top
    V_vip  = BETAS['B_VIP']      + bp * (p_vip  / 1000.0) + r
    V_prem = BETAS['B_PREMIUM']  + bp * (p_prem / 1000.0) + r
    V_est  = BETAS['B_ESTANDAR'] + bp * (p_est  / 1000.0) + r
    V_econ =                       bp * (p_econ / 1000.0) + r   # zona base
    V_none = BETAS['ASC_NINGUNO']
    return np.array([V_vip, V_prem, V_est, V_econ, V_none])

def simular(p_vip, p_prem, p_est, p_econ, rival_top):
    V = utilidades(p_vip, p_prem, p_est, p_econ, rival_top)
    exp_v = np.exp(V)
    probs = exp_v / exp_v.sum()
    zonas = ['VIP', 'Premium', 'Estándar', 'Económica', 'No compra']
    cuotas = dict(zip(zonas, probs))

    # Demanda teorica -> topada por capacidad fisica de cada zona
    precios = {'VIP': p_vip, 'Premium': p_prem, 'Estándar': p_est, 'Económica': p_econ}
    asistencia = {}
    for z in ['VIP', 'Premium', 'Estándar', 'Económica']:
        deseo = int(cuotas[z] * TOTAL_ASIENTOS)
        asistencia[z] = min(deseo, CAPACIDAD[z])

    asist_total = sum(asistencia.values())
    no_asisten = TOTAL_ASIENTOS - asist_total
    ingreso = sum(asistencia[z] * precios[z] for z in asistencia)

    # Cuotas REALES sobre aforo (lo que de verdad se vende; el sobrante va a "no asiste")
    cuotas_reales = {z: asistencia[z] / TOTAL_ASIENTOS * 100 for z in asistencia}
    cuotas_reales['No asiste / sin cupo'] = no_asisten / TOTAL_ASIENTOS * 100

    # Elasticidad-precio propia (formula Logit): beta_precio * (precio/1000) * (1 - P)
    elasticidades = {}
    for z in ['VIP', 'Premium', 'Estándar', 'Económica']:
        elasticidades[z] = BETAS['B_PRECIO'] * (precios[z] / 1000.0) * (1 - cuotas[z])

    return cuotas, cuotas_reales, asistencia, asist_total, no_asisten, ingreso, elasticidades

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Escenario del partido")
rival = st.sidebar.radio("Tipo de rival", ["Rival Top (alta convocatoria)", "Rival regular"])
rival_top = 1 if "Top" in rival else 0

st.sidebar.caption("ℹ️ El modelo estimó el efecto del rival como Top vs. No-Top. "
                   "Día de la semana y paquetes de alimentos NO se estimaron y no se incluyen.")

st.sidebar.header("💵 Precios por zona (MXN)")
st.sidebar.caption("Los rangos corresponden a lo evaluado en la encuesta.")
p_vip  = st.sidebar.slider("VIP",       *RANGOS['VIP'],       3800, step=100)
p_prem = st.sidebar.slider("Premium",   *RANGOS['Premium'],   2400, step=50)
p_est  = st.sidebar.slider("Estándar",  *RANGOS['Estándar'],  580,  step=20)
p_econ = st.sidebar.slider("Económica", *RANGOS['Económica'], 180,  step=10)

(cuotas, cuotas_reales, asistencia, asist_total,
 no_asisten, ingreso, elasticidades) = simular(p_vip, p_prem, p_est, p_econ, rival_top)

# =========================================================
# PANEL DE INDICADORES
# =========================================================
c1, c2, c3 = st.columns(3)
c1.metric("💰 Ingreso de taquilla", f"${ingreso/1e6:.2f}M MXN")
c2.metric("👥 Asistencia", f"{asist_total:,} / {TOTAL_ASIENTOS:,}",
          f"{asist_total/TOTAL_ASIENTOS*100:.1f}% de ocupación")
c3.metric("📉 No asiste / sin cupo", f"{no_asisten:,} fans")

st.markdown("---")

# =========================================================
# GRAFICA 1: Ocupacion fisica vs capacidad (barras agrupadas, compactas)
# =========================================================
st.subheader("🏟️ Ocupación física por zona")
df_oc = pd.DataFrame({
    'Zona': ['VIP', 'Premium', 'Estándar', 'Económica'],
    'Asientos vendidos': [asistencia[z] for z in ['VIP','Premium','Estándar','Económica']],
    'Capacidad máxima':  [CAPACIDAD[z]  for z in ['VIP','Premium','Estándar','Económica']],
})
df_oc_long = df_oc.melt(id_vars='Zona', var_name='Métrica', value_name='Asientos')

chart_oc = alt.Chart(df_oc_long).mark_bar().encode(
    x=alt.X('Métrica:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y('Asientos:Q', title='Asientos'),
    color=alt.Color('Métrica:N',
                    scale=alt.Scale(domain=['Asientos vendidos','Capacidad máxima'],
                                    range=['#C8102E', '#1D1D1B']),
                    legend=alt.Legend(title=None, orient='top')),
    column=alt.Column('Zona:N', title=None, header=alt.Header(labelOrient='bottom'))
).properties(width=120, height=260)
st.altair_chart(chart_oc, use_container_width=False)

st.markdown("---")

# =========================================================
# GRAFICA 2: Participacion de mercado (cuotas reales)
# =========================================================
st.subheader("📊 Distribución de la demanda (%)")
df_cuotas = pd.DataFrame({
    'Categoría': list(cuotas_reales.keys()),
    'Porcentaje': list(cuotas_reales.values()),
}).set_index('Categoría')
st.bar_chart(df_cuotas, use_container_width=True)

st.markdown("---")

# =========================================================
# ELASTICIDAD-PRECIO (derivada del beta real)
# =========================================================
st.subheader("📈 Elasticidad-precio por zona (escenario actual)")
st.caption("Elasticidad propia = β_precio × (precio/1000) × (1 − cuota). "
           "Valores entre 0 y −1 = inelástico (tolera subidas); menor a −1 = elástico (sensible).")

df_elast = pd.DataFrame({
    'Zona': list(elasticidades.keys()),
    'Precio actual': [f"${v:,}" for v in [p_vip, p_prem, p_est, p_econ]],
    'Elasticidad': [round(elasticidades[z], 2) for z in elasticidades],
    'Lectura': ['Elástica (sensible)' if elasticidades[z] < -1 else 'Inelástica (tolera precio)'
                for z in elasticidades],
})
st.dataframe(df_elast, use_container_width=True, hide_index=True)

st.markdown("---")

# =========================================================
# MATRIZ DE PRECIOS: evalua combinaciones dentro del rango seguro
# y rankea por ingreso de taquilla (con tope de capacidad)
# =========================================================
st.subheader("🧮 Matriz de precios — combinaciones óptimas")
st.caption("Evalúa 81 combinaciones (3 niveles × 4 zonas) dentro del rango de la encuesta, "
           "con el rival seleccionado, y las ordena por ingreso de taquilla.")

def construir_matriz(rival_top):
    niveles = {
        'VIP':       [2499, 3749, 4999],
        'Premium':   [1499, 2499, 3499],
        'Estándar':  [349, 574, 799],
        'Económica': [49, 174, 299],
    }
    filas = []
    for pv, pp, pe, pc in itertools.product(niveles['VIP'], niveles['Premium'],
                                            niveles['Estándar'], niveles['Económica']):
        _, _, asis, asis_tot, _, ing, elas = simular(pv, pp, pe, pc, rival_top)
        filas.append({
            'VIP': pv, 'Premium': pp, 'Estándar': pe, 'Económica': pc,
            'Ingreso ($M)': round(ing / 1e6, 2),
            'Asistencia': asis_tot,
            'Ocupación %': round(asis_tot / TOTAL_ASIENTOS * 100, 1),
        })
    return pd.DataFrame(filas).sort_values('Ingreso ($M)', ascending=False).reset_index(drop=True)

df_matriz = construir_matriz(rival_top)
st.dataframe(df_matriz.head(10), use_container_width=True, hide_index=True)

mejor = df_matriz.iloc[0]
st.success(
    f"**Combinación de mayor ingreso de taquilla:** "
    f"VIP ${int(mejor['VIP'])} · Premium ${int(mejor['Premium'])} · "
    f"Estándar ${int(mejor['Estándar'])} · Económica ${int(mejor['Económica'])} "
    f"→ ${mejor['Ingreso ($M)']}M con {mejor['Ocupación %']}% de ocupación."
)
st.caption("⚠️ Maximiza ingreso de **taquilla**. El club prioriza aforo porque el ingreso "
           "secundario (alimentos, mercancía, patrocinios) depende del volumen de asistentes; "
           "considera ese objetivo al elegir entre filas de ingreso similar pero distinta ocupación.")

st.markdown("---")

# =========================================================
# NOTA METODOLOGICA (defendible ante el board / profesores)
# =========================================================
with st.expander("📋 Nota metodológica — qué hay detrás de estos números"):
    st.markdown(f"""
- **Modelo:** Choice-Based Conjoint estimado en Biogeme (`HB_Capitanes_Real_4Zonas`),
  {MODELO_INFO['n_obs']} elecciones observadas.
- **Validación:** {MODELO_INFO['rho2_nota']}.
- **Atributos estimados:** Zona (VIP / Premium / Estándar / Económica-base),
  Precio y Rival (Top vs. No-Top).
- **Disposición a pagar (WTP) estimada:** VIP $676 · Premium $646 · Estándar $1,650 · Rival Top $484.
  La zona Estándar concentra la mayor disposición a pagar.
- **Alcance:** se usan las **medias poblacionales** del modelo. La heterogeneidad individual
  (componente jerárquico/Bayes) no convergió con 1,000 draws y queda como trabajo futuro;
  requiere más draws de Monte Carlo.
- **No incluido:** día de la semana y paquetes de alimentos/bebidas **no se estimaron**
  en este modelo, por lo que no forman parte del simulador.
- **Rango seguro:** los precios se mantienen dentro del rango evaluado en la encuesta;
  fuera de él el modelo extrapolaría sin respaldo de datos.
""")
