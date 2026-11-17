import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Rendimiento Agrícola", page_icon="🌿", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=":memory:")
    con.execute("SELECT setseed(0.42);")
    con.execute("""
        CREATE TABLE base AS
        SELECT
            i AS parcela_id,
            ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
            ROUND(50.0  + random() * 100.0, 2) AS fertilizante_kg,
            ROUND(18.0  + random() * 14.0, 2)  AS temperatura_c,
            random()                           AS u_trat,
            (random() * 1.6 - 0.8)             AS ruido
        FROM range(1, 101) t(i);
    """)
    con.execute("""
        CREATE TABLE telemetria AS
        SELECT parcela_id, volumen_riego, fertilizante_kg, temperatura_c,
            CASE WHEN u_trat < 0.5 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            ROUND(5.0 + 0.035 * volumen_riego + 0.020 * fertilizante_kg
                  + CASE WHEN u_trat >= 0.5 THEN 1.8 ELSE 0.0 END + ruido, 2) AS rendimiento_ton
        FROM base;
    """)
    df = con.execute("SELECT * FROM telemetria ORDER BY parcela_id").df()
    con.close()
    return df


df_total = cargar_datos()

# Filtros dinámicos (barra lateral)
st.sidebar.header("🎛️ Filtros")
trats = st.sidebar.multiselect(
    "Tipo de tratamiento",
    ["Tratamiento_A", "Tratamiento_B"],
    default=["Tratamiento_A", "Tratamiento_B"],
)
r_min, r_max = st.sidebar.slider("Volumen de riego (m³/ha)", 100.0, 300.0, (100.0, 300.0))

df = df_total[
    df_total["tipo_tratamiento"].isin(trats)
    & df_total["volumen_riego"].between(r_min, r_max)
].copy()

st.title("🌿 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")


if len(df) < 10:
    st.warning("Hay muy pocos datos con estos filtros. Amplía el rango o selecciona más tratamientos.")
    st.stop()


alpha = 0.05

# Modelo OLS
X = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])
modelo = sm.OLS(df["rendimiento_ton"], X).fit()
df["residuo"] = df["rendimiento_ton"] - modelo.predict(X)

# Prueba t (H1: mu_B - mu_A > 0)
A = df.loc[df["tipo_tratamiento"] == "Tratamiento_A", "rendimiento_ton"]
B = df.loc[df["tipo_tratamiento"] == "Tratamiento_B", "rendimiento_ton"]
hay_ambos = len(A) > 1 and len(B) > 1

if hay_ambos:
    t_stat, p_valor = stats.ttest_ind(B, A, equal_var=True, alternative="greater")
    gl = len(A) + len(B) - 2
    sp = np.sqrt(((len(A) - 1) * A.var(ddof=1) + (len(B) - 1) * B.var(ddof=1)) / gl)
    dif = B.mean() - A.mean()
    err = sp * np.sqrt(1 / len(A) + 1 / len(B))
    t_crit = stats.t.ppf(0.975, gl)
    ic_inf, ic_sup = dif - t_crit * err, dif + t_crit * err

# 3 tarjetas KPI
k1, k2, k3 = st.columns(3)
k1.metric("Rendimiento promedio global", f"{df['rendimiento_ton'].mean():.2f} ton/ha")
k2.metric("R² ajustado del modelo", f"{modelo.rsquared_adj * 100:.2f} %")
k3.metric("P-valor prueba t (B > A)", f"{p_valor:.3g}" if hay_ambos else "N/D")

# Veredicto
if hay_ambos:
    if p_valor < alpha:
        st.success(f"✅ p = {p_valor:.3g} < {alpha}: se rechaza H₀. El Tratamiento B incrementa el rendimiento "
                   f"({dif:.2f} ton/ha, IC 95% [{ic_inf:.2f}, {ic_sup:.2f}]). Se recomienda autorizar la inversión.")
    else:
        st.error(f"❌ p = {p_valor:.3g} ≥ {alpha}: no se rechaza H₀. No hay evidencia para escalar el Tratamiento B.")
else:
    st.info("Selecciona ambos tratamientos para ejecutar la prueba de hipótesis.")

st.divider()


fila1 = st.columns(2)
fila2 = st.columns(2)

# Gráfica 1: Heatmap
with fila1[0]:
    st.subheader("1. Mapa de calor de correlación")
    cols = ["volumen_riego", "fertilizante_kg", "temperatura_c", "rendimiento_ton"]
    etiquetas = ["Riego", "Fertilizante", "Temperatura", "Rendimiento"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
                xticklabels=etiquetas, yticklabels=etiquetas, ax=ax)
    st.pyplot(fig)

# Gráfica 2: Dispersión + OLS
with fila1[1]:
    st.subheader("2. Regresión OLS: Rendimiento vs. Riego")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", ci=95,
                scatter_kws={"alpha": 0.6, "s": 25}, line_kws={"color": "crimson"}, ax=ax)
    ax.set_xlabel("Volumen de riego (m³/ha)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig)

# Gráfica 3: Boxplot
with fila2[0]:
    st.subheader("3. Boxplot de tratamientos")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    orden = [t for t in ["Tratamiento_A", "Tratamiento_B"] if t in df["tipo_tratamiento"].unique()]
    sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", order=orden,
                hue="tipo_tratamiento", palette="Set2", legend=False, ax=ax)
    medias = df.groupby("tipo_tratamiento")["rendimiento_ton"].mean().reindex(orden)
    ax.scatter(range(len(orden)), medias.values, marker="D", color="black", zorder=5, label="Media")
    ax.set_xlabel("Tipo de tratamiento (A vs B)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    ax.legend()
    st.pyplot(fig)
    if hay_ambos:
        st.caption(f"Dif. de medias (B − A) = {dif:.2f} ton/ha · IC 95% [{ic_inf:.2f}, {ic_sup:.2f}]")

# Gráfica 4: Residuos
with fila2[1]:
    st.subheader("4. Histograma de residuos")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.histplot(df["residuo"], kde=True, bins=15, color="steelblue", ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residuo = Y real − Y predicho (ton/ha)")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

with st.expander("📄 Resumen estadístico OLS"):
    st.text(modelo.summary().as_text())

