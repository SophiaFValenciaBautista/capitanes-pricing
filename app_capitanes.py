import streamlit as st
import pandas as pd
import numpy as np
import itertools
import altair as alt

# =========================================================
# SIMULADOR DE PRICING — CAPITANES CDMX
# Modelo: Choice-Based Conjoint estimado en Biogeme
# (HB_Capitanes_Ampliado, medias poblacionales).
# Dataset definitivo: 1,170 elecciones. TODOS los betas son reales.
# Variables marcadas con su significancia estadistica (95%).
# =========================================================

st.set_page_config(page_title="Capitanes CDMX Pricing", layout="wide", page_icon="🏀")

st.title("🏀 Capitanes CDMX — Simulador de Pricing")
st.markdown("""
Simulador de demanda e ingresos de taquilla para la Arena CDMX, basado en un modelo
**Choice-Based Conjoint estimado en Biogeme**. Todos los coeficientes provienen del modelo real;
cada atributo está marcado con su **significancia estadística** al 95%.
""")

# ---------------------------------------------------------
# BETAS REALES — modelo HB_Capitanes_Ampliado (1,170 elecciones)
# Escala: precio dividido entre 1000. Zona base = Económica.
# Rival base = Austin Spurs. Día base = Lun–Jue. F&B solo en VIP/Premium.
# ---------------------------------------------------------
BETAS = {
    'B_VIP':           0.21835811950929115,
    'B_PREMIUM':       0.18473681471448933,
    'B_ESTANDAR':      0.8464164862496442,
    'B_PRECIO':       -0.5190885553010625,
    'B_RIVAL_LAKERS':  0.2342081604588969,
    'B_RIVAL_HUSTLE':  0.039138129439401474,
    'B_DIA_FINDE':     0.031126454169117863,
    'B_FB':            0.23008637722174097,
    'ASC_NINGUNO':    -1.0367189600656634,
}

# Significancia (p-valores robustos). True = significativo al 95%.
SIGNIF = {
    'Precio':     (0.00002, True),
    'Estándar':   (0.00008, True),
    'No-compra':  (0.00006, True),
    'Lakers':     (0.113,   False),
    'F&B':        (0.177,   False),
    'Premium':    (0.539,   False),
    'VIP':        (0.639,   False),
    'Día finde':  (0.823,   False),
    'Hustle':     (0.751,   False),
}

# Inventario real Arena CDMX (18,037)
CAPACIDAD = {'VIP': 1732, 'Premium': 3556, 'Estándar': 6373, 'Económica': 6376}
TOTAL_ASIENTOS = sum(CAPACIDAD.values())

# Rangos de precio evaluados en la encuesta (zona segura)
RANGOS = {
    'VIP': (2499, 4999), 'Premium': (1499, 3499),
    'Estándar': (349, 799), 'Económica': (49, 299),
}

# =========================================================
# MOTOR — Logit con betas reales, sin factores de ajuste
# =========================================================
def _efecto_rival(rival):
    # 0 = Austin Spurs (base), 1 = Southbay Lakers, 2 = Memphis Hustle
    if rival == 1:
        return BETAS['B_RIVAL_LAKERS']
    if rival == 2:
        return BETAS['B_RIVAL_HUSTLE']
    return 0.0

def simular(p_vip, p_prem, p_est, p_econ, rival, finde, fb_on):
    bp = BETAS['B_PRECIO']
    r = _efecto_rival(rival)
    dia = BETAS['B_DIA_FINDE'] * finde
    fb = BETAS['B_FB'] * fb_on          # F&B solo se suma a VIP y Premium
    V = np.array([
        BETAS['B_VIP']      + bp * (p_vip  / 1000.0) + r + dia + fb,
        BETAS['B_PREMIUM']  + bp * (p_prem / 1000.0) + r + dia + fb,
        BETAS['B_ESTANDAR'] + bp * (p_est  / 1000.0) + r + dia,
        bp * (p_econ / 1000.0) + r + dia,             # Económica (base)
        BETAS['ASC_NINGUNO'],
    ])
    probs = np.exp(V) / np.exp(V).sum()
    zonas = ['VIP', 'Premium', 'Estándar', 'Económica', 'No compra']
    cuotas = dict(zip(zonas, probs))

    precios = {'VIP': p_vip, 'Premium': p_prem, 'Estándar': p_est, 'Económica': p_econ}
    asistencia = {z: min(int(cuotas[z] * TOTAL_ASIENTOS), CAPACIDAD[z]) for z in CAPACIDAD}
    asist_total = sum(asistencia.values())
    no_asisten = TOTAL_ASIENTOS - asist_total
    ingreso = sum(asistencia[z] * precios[z] for z in CAPACIDAD)

    cuotas_reales = {z: asistencia[z] / TOTAL_ASIENTOS * 100 for z in CAPACIDAD}
    cuotas_reales['No asiste / sin cupo'] = no_asisten / TOTAL_ASIENTOS * 100

    elasticidades = {z: BETAS['B_PRECIO'] * (precios[z] / 1000.0) * (1 - cuotas[z]) for z in CAPACIDAD}
    return cuotas, cuotas_reales, asistencia, asist_total, no_asisten, ingreso, elasticidades

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Escenario del partido")

rival_lbl = st.sidebar.selectbox(
    "Rival",
    ["Austin Spurs (base)", "Southbay Lakers", "Memphis Hustle"],
)
rival = {"Austin Spurs (base)": 0, "Southbay Lakers": 1, "Memphis Hustle": 2}[rival_lbl]

dia_lbl = st.sidebar.radio("Día", ["Entre semana (Lun–Jue)", "Fin de semana (Vie–Dom)"])
finde = 1 if "Fin" in dia_lbl else 0

fb_lbl = st.sidebar.radio("Paquete de alimentos y bebidas", ["Sin paquete", "Con paquete F&B"])
fb_on = 1 if "Con" in fb_lbl else 0
st.sidebar.caption("ℹ️ El paquete F&B solo aplica a VIP y Premium (así se diseñó la encuesta).")

st.sidebar.markdown("---")
st.sidebar.header("💵 Precios por zona (MXN)")
st.sidebar.caption("Rangos = lo evaluado en la encuesta.")
p_vip  = st.sidebar.slider("VIP",       *RANGOS['VIP'],       3800, step=100)
p_prem = st.sidebar.slider("Premium",   *RANGOS['Premium'],   2400, step=50)
p_est  = st.sidebar.slider("Estándar",  *RANGOS['Estándar'],  580,  step=20)
p_econ = st.sidebar.slider("Económica", *RANGOS['Económica'], 180,  step=10)

(cuotas, cuotas_reales, asistencia, asist_total,
 no_asisten, ingreso, elasticidades) = simular(p_vip, p_prem, p_est, p_econ, rival, finde, fb_on)

# =========================================================
# AVISO DE SIGNIFICANCIA (opción 1)
# =========================================================
st.info(
    "📌 **Lectura estadística:** en el modelo, solo el **Precio** y la **zona Estándar** "
    "resultaron significativos al 95%. El rival, el día y el paquete F&B se incluyen como "
    "**controles exploratorios** (no significativos en la muestra actual). Sus efectos se "
    "muestran, pero deben interpretarse con cautela."
)

# =========================================================
# INDICADORES
# =========================================================
c1, c2, c3 = st.columns(3)
c1.metric("💰 Ingreso de taquilla", f"${ingreso/1e6:.2f}M MXN")
c2.metric("👥 Asistencia", f"{asist_total:,} / {TOTAL_ASIENTOS:,}",
          f"{asist_total/TOTAL_ASIENTOS*100:.1f}% ocupación")
c3.metric("📉 No asiste / sin cupo", f"{no_asisten:,} fans")

st.markdown("---")

# =========================================================
# GRÁFICA 1 — ocupación física vs capacidad
# =========================================================
st.subheader("🏟️ Ocupación física por zona")
df_oc = pd.DataFrame({
    'Zona': ['VIP', 'Premium', 'Estándar', 'Económica'],
    'Asientos vendidos': [asistencia[z] for z in ['VIP','Premium','Estándar','Económica']],
    'Capacidad máxima':  [CAPACIDAD[z]  for z in ['VIP','Premium','Estándar','Económica']],
}).melt(id_vars='Zona', var_name='Métrica', value_name='Asientos')

chart_oc = alt.Chart(df_oc).mark_bar().encode(
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
# GRÁFICA 2 — distribución de demanda
# =========================================================
st.subheader("📊 Distribución de la demanda (%)")
df_cuotas = pd.DataFrame({
    'Categoría': list(cuotas_reales.keys()),
    'Porcentaje': list(cuotas_reales.values()),
}).set_index('Categoría')
st.bar_chart(df_cuotas, use_container_width=True)

st.markdown("---")

# =========================================================
# ELASTICIDAD
# =========================================================
st.subheader("📈 Elasticidad-precio por zona (escenario actual)")
st.caption("Elasticidad propia = β_precio × (precio/1000) × (1 − cuota). "
           "Entre 0 y −1 = inelástica; menor a −1 = elástica.")
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
# MATRIZ DE PRECIOS
# =========================================================
st.subheader("🧮 Matriz de precios — combinaciones óptimas")
st.caption("81 combinaciones (3 niveles × 4 zonas) dentro del rango de la encuesta, "
           "con el escenario seleccionado (rival, día, F&B), ordenadas por ingreso de taquilla.")

def construir_matriz(rival, finde, fb_on):
    niveles = {
        'VIP': [2499, 3749, 4999], 'Premium': [1499, 2499, 3499],
        'Estándar': [349, 574, 799], 'Económica': [49, 174, 299],
    }
    filas = []
    for pv, pp, pe, pc in itertools.product(niveles['VIP'], niveles['Premium'],
                                            niveles['Estándar'], niveles['Económica']):
        _, _, _, asis_tot, _, ing, _ = simular(pv, pp, pe, pc, rival, finde, fb_on)
        filas.append({
            'VIP': pv, 'Premium': pp, 'Estándar': pe, 'Económica': pc,
            'Ingreso ($M)': round(ing / 1e6, 2),
            'Asistencia': asis_tot,
            'Ocupación %': round(asis_tot / TOTAL_ASIENTOS * 100, 1),
        })
    return pd.DataFrame(filas).sort_values('Ingreso ($M)', ascending=False).reset_index(drop=True)

df_matriz = construir_matriz(rival, finde, fb_on)
st.dataframe(df_matriz.head(10), use_container_width=True, hide_index=True)

mejor = df_matriz.iloc[0]
st.success(
    f"**Combinación de mayor ingreso de taquilla:** "
    f"VIP ${int(mejor['VIP'])} · Premium ${int(mejor['Premium'])} · "
    f"Estándar ${int(mejor['Estándar'])} · Económica ${int(mejor['Económica'])} "
    f"→ ${mejor['Ingreso ($M)']}M con {mejor['Ocupación %']}% de ocupación."
)
st.caption("⚠️ Maximiza ingreso de taquilla. El club prioriza aforo porque el ingreso "
           "secundario (alimentos, mercancía, patrocinios) depende del volumen; pondéralo "
           "al elegir entre filas de ingreso similar y distinta ocupación.")

st.markdown("---")

# =========================================================
# NOTA METODOLÓGICA
# =========================================================
with st.expander("📋 Nota metodológica y significancia estadística"):
    st.markdown("""
**Modelo:** Choice-Based Conjoint estimado en Biogeme (`HB_Capitanes_Ampliado`),
1,170 elecciones observadas. Se usan las medias poblacionales (la heterogeneidad
individual no convergió con 1,000 draws; queda como trabajo futuro).

**Validación:** accuracy holdout ≈ 45% (azar con 4 opciones = 25%).

**Significancia de cada atributo (p-valor robusto, 95%):**

| Atributo | p-valor | ¿Significativo? |
|---|---|---|
| Precio | 0.00002 | ✅ Sí |
| Zona Estándar | 0.00008 | ✅ Sí |
| Constante No-compra | 0.00006 | ✅ Sí |
| Rival Lakers | 0.113 | ❌ No |
| Paquete F&B | 0.177 | ❌ No |
| Zona Premium | 0.539 | ❌ No |
| Zona VIP | 0.639 | ❌ No |
| Rival Hustle | 0.751 | ❌ No |
| Día fin de semana | 0.823 | ❌ No |

**Interpretación de negocio:** para los fans de Capitanes en esta muestra, el **precio**
y el **salto a la zona Estándar** son los únicos drivers estadísticamente determinantes.
El día de la semana, el paquete F&B y el rival específico no mostraron un efecto
significativo al 95%; confirmarlos requeriría ampliar el tamaño de muestra en el fieldwork.
El resultado se mantiene estable al añadir más respuestas, lo que indica robustez y no
un artefacto de muestreo.

**WTP estimada (MXN):** VIP $421 · Premium $356 · Estándar $1,631 · Rival Lakers $451 ·
Rival Hustle $75 · Día finde $60 · Paquete F&B $443.

**No extrapolar:** los precios se mantienen dentro del rango evaluado en la encuesta.
""")
