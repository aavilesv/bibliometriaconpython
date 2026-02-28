import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# 1) Parámetros de simulación
# -----------------------------
N = 220               # tamaño muestral
SEED = 42
MISSING_RATE = 0.03   # % faltantes en ítems Likert
EFFECT_VI_TO_VD = 0.55  # fuerza relación VI -> VD
NOISE_ITEM = 0.60     # ruido por ítem (↑ => alfa ↓)
NOISE_DIM = 0.45      # ruido por dimensión (↑ => alfa ↓)

rng = np.random.default_rng(SEED)

# -----------------------------
# 2) Catálogos (perfil)
# -----------------------------
areas = ["Bodega", "Transporte", "Coordinación", "Compras", "TI/Analítica", "Operaciones"]
cargos = ["Operativo", "Analista", "Coordinación", "Supervisión", "Jefatura"]
turnos = ["Mañana", "Tarde", "Noche", "Rotativo"]

def sample_categorical(options, p=None, size=N):
    return rng.choice(options, size=size, replace=True, p=p)

area = sample_categorical(areas, p=[0.34, 0.22, 0.18, 0.10, 0.06, 0.10])
cargo = sample_categorical(cargos, p=[0.45, 0.22, 0.12, 0.15, 0.06])
turno = sample_categorical(turnos, p=[0.35, 0.35, 0.10, 0.20])

edad = np.clip(rng.normal(loc=33, scale=8.5, size=N).round().astype(int), 18, 60)
antiguedad = np.clip(rng.gamma(shape=2.2, scale=1.6, size=N), 0, 15)
antiguedad = np.round(antiguedad, 1)

contacto = sample_categorical(["Bajo", "Medio", "Alto"], p=[0.25, 0.50, 0.25])

# -----------------------------
# 3) Latentes por dimensión (realismo)
# -----------------------------
area_effect = pd.Series(area).map({
    "TI/Analítica": 0.50,
    "Coordinación": 0.20,
    "Operaciones": 0.10,
    "Bodega": 0.05,
    "Transporte": 0.00,
    "Compras": -0.05
}).to_numpy()

contact_effect = pd.Series(contacto).map({"Bajo": -0.35, "Medio": 0.00, "Alto": 0.35}).to_numpy()

vi_lat = rng.normal(0, 1, N) + area_effect + contact_effect + rng.normal(0, NOISE_DIM, N)

td_at_lat = 0.75 * vi_lat + rng.normal(0, NOISE_DIM, N)  # adopción tecnológica
td_cd_lat = 0.65 * vi_lat + rng.normal(0, NOISE_DIM, N)  # capacidades digitales

vd_lat = EFFECT_VI_TO_VD * vi_lat + rng.normal(0, 1.0, N)

ol_eo_lat = 0.70 * vd_lat + rng.normal(0, NOISE_DIM, N)  # eficiencia operativa
ol_iv_lat = 0.68 * vd_lat + rng.normal(0, NOISE_DIM, N)  # integración/visibilidad

# -----------------------------
# 4) Conversión a Likert 1-5
# -----------------------------
def to_likert(x, thresholds=(-1.2, -0.3, 0.3, 1.2)):
    t1, t2, t3, t4 = thresholds
    return np.select(
        [x < t1, (x >= t1) & (x < t2), (x >= t2) & (x < t3), (x >= t3) & (x < t4), x >= t4],
        [1, 2, 3, 4, 5]
    ).astype(int)

def make_items(latent, n_items, prefix):
    items = {}
    for i in range(1, n_items + 1):
        item_bias = rng.normal(0, 0.08)
        cont = latent + rng.normal(0, NOISE_ITEM, N) + item_bias
        items[f"{prefix}{i}"] = to_likert(cont)
    return pd.DataFrame(items)

df_td_at = make_items(td_at_lat, 4, "TD_AT")  # TD_AT1..4
df_td_cd = make_items(td_cd_lat, 4, "TD_CD")  # TD_CD1..4
df_ol_eo = make_items(ol_eo_lat, 4, "OL_EO")  # OL_EO1..4
df_ol_iv = make_items(ol_iv_lat, 4, "OL_IV")  # OL_IV1..4

# -----------------------------
# 5) DataFrame final
# -----------------------------
df = pd.DataFrame({
    "edad": edad,
    "area": area,
    "cargo": cargo,
    "turno": turno,
    "antiguedad_anios": antiguedad,
    "contacto_sistemas": contacto,
})

df = pd.concat([df, df_td_at, df_td_cd, df_ol_eo, df_ol_iv], axis=1)

likert_cols = [c for c in df.columns if c.startswith(("TD_AT", "TD_CD", "OL_EO", "OL_IV"))]
mask = rng.random((N, len(likert_cols))) < MISSING_RATE
df.loc[:, likert_cols] = df.loc[:, likert_cols].mask(mask)

# -----------------------------
# 6) Alfa de Cronbach
# -----------------------------
def cronbach_alpha(data: pd.DataFrame) -> float:
    x = data.copy()
    x = x.apply(lambda col: col.fillna(col.mean()), axis=0)  # imputación por media

    k = x.shape[1]
    if k < 2:
        return np.nan

    item_vars = x.var(axis=0, ddof=1)
    total_score = x.sum(axis=1)
    total_var = total_score.var(ddof=1)
    if total_var == 0:
        return np.nan

    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)

items_td_at = [f"TD_AT{i}" for i in range(1, 5)]
items_td_cd = [f"TD_CD{i}" for i in range(1, 5)]
items_ol_eo = [f"OL_EO{i}" for i in range(1, 5)]
items_ol_iv = [f"OL_IV{i}" for i in range(1, 5)]
items_total = items_td_at + items_td_cd + items_ol_eo + items_ol_iv

alpha_total = cronbach_alpha(df[items_total])
alpha_td_at = cronbach_alpha(df[items_td_at])
alpha_td_cd = cronbach_alpha(df[items_td_cd])
alpha_ol_eo = cronbach_alpha(df[items_ol_eo])
alpha_ol_iv = cronbach_alpha(df[items_ol_iv])

# -----------------------------
# 7) Índices + hallazgos básicos
# -----------------------------
def scale_mean(df_in, cols):
    return df_in[cols].astype(float).mean(axis=1)

df["VI_TD_AT"] = scale_mean(df, items_td_at)
df["VI_TD_CD"] = scale_mean(df, items_td_cd)
df["VI_TD_TOTAL"] = df[["VI_TD_AT", "VI_TD_CD"]].mean(axis=1)

df["VD_OL_EO"] = scale_mean(df, items_ol_eo)
df["VD_OL_IV"] = scale_mean(df, items_ol_iv)
df["VD_OL_TOTAL"] = df[["VD_OL_EO", "VD_OL_IV"]].mean(axis=1)

# Correlación Spearman (recomendable con Likert)
corr_spearman = df[["VI_TD_TOTAL", "VD_OL_TOTAL"]].corr(method="spearman").iloc[0, 1]

# Regresión simple aproximada: VD = b0 + b1*VI
x = df["VI_TD_TOTAL"].to_numpy()
y = df["VD_OL_TOTAL"].to_numpy()
x_mean, y_mean = np.mean(x), np.mean(y)
b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b0 = y_mean - b1 * x_mean
y_pred = b0 + b1 * x
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r2 = 1 - ss_res / ss_tot

# Descriptivos por dimensión
desc = pd.DataFrame({
    "dimension": ["Adopción tecnológica", "Capacidades digitales", "Eficiencia operativa", "Integración/visibilidad", "VI total", "VD total"],
    "media": [
        df["VI_TD_AT"].mean(),
        df["VI_TD_CD"].mean(),
        df["VD_OL_EO"].mean(),
        df["VD_OL_IV"].mean(),
        df["VI_TD_TOTAL"].mean(),
        df["VD_OL_TOTAL"].mean()
    ],
    "desv_est": [
        df["VI_TD_AT"].std(ddof=1),
        df["VI_TD_CD"].std(ddof=1),
        df["VD_OL_EO"].std(ddof=1),
        df["VD_OL_IV"].std(ddof=1),
        df["VI_TD_TOTAL"].std(ddof=1),
        df["VD_OL_TOTAL"].std(ddof=1)
    ]
})

# Promedios por área (útil para hallazgos)
by_area = df.groupby("area")[["VI_TD_TOTAL", "VD_OL_TOTAL", "VI_TD_AT", "VI_TD_CD", "VD_OL_EO", "VD_OL_IV"]].mean().reset_index()

# Tabla de alfas
alphas = pd.DataFrame({
    "escala": [
        "Total (16 ítems)",
        "Adopción tecnológica (4 ítems)",
        "Capacidades digitales (4 ítems)",
        "Eficiencia operativa (4 ítems)",
        "Integración/visibilidad (4 ítems)"
    ],
    "alpha": [alpha_total, alpha_td_at, alpha_td_cd, alpha_ol_eo, alpha_ol_iv]
})

# Resumen de relación VI->VD
relacion = pd.DataFrame([{
    "correlacion_spearman_VI_VD": corr_spearman,
    "regresion_b0": b0,
    "regresion_b1": b1,
    "R2": r2,
    "N": N,
    "missing_rate": MISSING_RATE
}])

# -----------------------------
# 8) Exportar a tu ruta en G:
# -----------------------------
output_dir = Path(r"G:\Mi unidad\Master en administración y empresas\Títtulación final\tarea_montoya")
output_dir.mkdir(parents=True, exist_ok=True)

csv_base = output_dir / "encuesta_simulada_transformacion_digital_logistica.csv"
csv_alphas = output_dir / "hallazgos_alfa_cronbach.csv"
csv_desc = output_dir / "hallazgos_descriptivos_dimensiones.csv"
csv_rel = output_dir / "hallazgos_relacion_VI_VD.csv"
csv_area = output_dir / "hallazgos_promedios_por_area.csv"

df.to_csv(csv_base, index=False, encoding="utf-8-sig")
alphas.to_csv(csv_alphas, index=False, encoding="utf-8-sig")
desc.to_csv(csv_desc, index=False, encoding="utf-8-sig")
relacion.to_csv(csv_rel, index=False, encoding="utf-8-sig")
by_area.to_csv(csv_area, index=False, encoding="utf-8-sig")

# -----------------------------
# 9) Imprimir hallazgos clave
# -----------------------------
print("\n=== Exportación ===")
print("Base simulada:", csv_base)
print("Alfa Cronbach:", csv_alphas)
print("Descriptivos:", csv_desc)
print("Relación VI-VD:", csv_rel)
print("Promedios por área:", csv_area)

print("\n=== Alfa de Cronbach (simulado) ===")
print(f"Total (16 ítems): {alpha_total:.3f}")
print(f"Adopción tecnológica (4): {alpha_td_at:.3f}")
print(f"Capacidades digitales (4): {alpha_td_cd:.3f}")
print(f"Eficiencia operativa (4): {alpha_ol_eo:.3f}")
print(f"Integración/visibilidad (4): {alpha_ol_iv:.3f}")

print("\n=== Descriptivos (Likert 1-5) ===")
print(desc.to_string(index=False))

print("\n=== Asociación VI->VD ===")
print(f"Spearman(VI, VD): {corr_spearman:.3f}")
print(f"Regresión: VD = {b0:.3f} + {b1:.3f}*VI | R² = {r2:.3f}")
