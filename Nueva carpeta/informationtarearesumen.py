# ============================================================
# 02_analizar_datos.py
# Analisis: indices, alfa, descriptivos, correlaciones, modelo y graficas
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 0) Ruta de trabajo (AJUSTA si hace falta)
# ----------------------------
BASE_DIR = Path(r"G:\Mi unidad\2025\codigos bibliometria NPL")
CSV_IN = BASE_DIR / "datos_simulados_fintech_milagro.csv"

OUT_DIR = BASE_DIR / "salidas_analisis"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) Cargar datos
# ----------------------------
if not CSV_IN.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo:\n{CSV_IN}\n"
        "Verifica que el CSV exista y que el nombre coincida."
    )

df = pd.read_csv(CSV_IN)

# ----------------------------
# 2) Validaciones minimas
# ----------------------------
required_cols = [
    "U1","U2","U3","U4","U5",
    "P1","P2","P3","P4","P5","P6",
    "A1","A2","A3","A4","A5",
    "S1","S2","S3","S4","S5",
    "acceso_internet_30d","nivel_instruccion","smartphone"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en el CSV: {missing}")

# ----------------------------
# 3) Funciones (Alfa de Cronbach)
# ----------------------------
def cronbach_alpha(data: pd.DataFrame) -> float:
    data = data.dropna()
    k = data.shape[1]
    if k < 2:
        return float("nan")
    item_vars = data.var(ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return (k/(k-1)) * (1 - item_vars.sum()/total_var)

# ----------------------------
# 4) Recodificaciones
# ----------------------------
# P6 inverso para analisis
df["P6_rev"] = df["P6"].map({1:5,2:4,3:3,4:2,5:1})

# Controles ordinales
internet_map = {"Nunca":0, "Rara vez":1, "Ocasional":2, "Frecuente":3, "Siempre":4}
edu_map = {"Primaria":0, "Secundaria":1, "Tecnica/tecnologica":2, "Universitaria":3, "Posgrado":4}

df["internet_score"] = df["acceso_internet_30d"].map(internet_map)
df["edu_score"] = df["nivel_instruccion"].map(edu_map)
df["smartphone_bin"] = (df["smartphone"].astype(str).str.lower() == "si").astype(int)

# ----------------------------
# 5) Indices por dimension (promedios)
# ----------------------------
df["VI_uso_tecnologias"] = df[["U1","U2","U3","U4","U5"]].mean(axis=1)
df["VI_percepcion_tecnologica"] = df[["P1","P2","P3","P4","P5","P6_rev"]].mean(axis=1)

df["VD_acceso_financiero"] = df[["A1","A2","A3","A4","A5"]].mean(axis=1)
df["VD_uso_servicios"] = df[["S1","S2","S3","S4","S5"]].mean(axis=1)

df["VD_inclusion_financiera"] = df[["VD_acceso_financiero","VD_uso_servicios"]].mean(axis=1)

index_cols = [
    "VI_uso_tecnologias",
    "VI_percepcion_tecnologica",
    "VD_acceso_financiero",
    "VD_uso_servicios",
    "VD_inclusion_financiera"
]

# ----------------------------
# 6) Confiabilidad (Alfa)
# ----------------------------
alpha_u = cronbach_alpha(df[["U1","U2","U3","U4","U5"]])
alpha_p = cronbach_alpha(df[["P1","P2","P3","P4","P5","P6_rev"]])
alpha_a = cronbach_alpha(df[["A1","A2","A3","A4","A5"]])
alpha_s = cronbach_alpha(df[["S1","S2","S3","S4","S5"]])

alpha_total = cronbach_alpha(df[
    ["U1","U2","U3","U4","U5",
     "P1","P2","P3","P4","P5","P6_rev",
     "A1","A2","A3","A4","A5",
     "S1","S2","S3","S4","S5"]
])

alpha_summary = pd.DataFrame({
    "Dimension": [
        "Uso FinTech (U1-U5)",
        "Percepcion (P1-P5 + P6 invertido)",
        "Acceso (A1-A5)",
        "Uso servicios (S1-S5)",
        "Instrumento total"
    ],
    "Alfa_Cronbach": [alpha_u, alpha_p, alpha_a, alpha_s, alpha_total]
})

# ----------------------------
# 7) Descriptivos y correlaciones (Spearman)
# ----------------------------
descriptivos = df[index_cols].describe().T
corr_spearman = df[index_cols].corr(method="spearman")

# ----------------------------
# 8) Modelo (OLS) con controles
# ----------------------------
# OLS es util para simulacion; con Likert se recomienda tambien robustez, pero esto cumple tarea.
try:
    import statsmodels.api as sm

    X = df[[
        "VI_uso_tecnologias",
        "VI_percepcion_tecnologica",
        "internet_score",
        "edu_score",
        "smartphone_bin"
    ]].copy()

    # Asegurar sin nulos en regresion
    reg_df = pd.concat([df["VD_inclusion_financiera"], X], axis=1).dropna()
    y = reg_df["VD_inclusion_financiera"]
    X = reg_df.drop(columns=["VD_inclusion_financiera"])

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    reg_info = pd.DataFrame({
        "Metrica": ["R2", "R2_ajustado", "N"],
        "Valor": [model.rsquared, model.rsquared_adj, int(model.nobs)]
    })

    reg_coef = model.summary2().tables[1].reset_index().rename(columns={"index":"Parametro"})

except Exception as e:
    # Fallback minimo si statsmodels no esta instalado
    reg_info = pd.DataFrame({
        "Metrica": ["Nota"],
        "Valor": [f"No se pudo ejecutar OLS con statsmodels: {e}"]
    })
    reg_coef = pd.DataFrame()

# ----------------------------
# 9) Graficas (matplotlib)
# ----------------------------

# 9.1 Histogramas por indice
for col in index_cols:
    plt.figure()
    plt.hist(df[col].dropna(), bins=10)
    plt.title(f"Distribucion del indice: {col}")
    plt.xlabel("Puntaje (1-5)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"hist_{col}.png", dpi=200)
    plt.close()

# 9.2 Dispersión: Uso FinTech vs Inclusion + linea
plt.figure()
x = df["VI_uso_tecnologias"].to_numpy()
y = df["VD_inclusion_financiera"].to_numpy()
mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]; y = y[mask]

plt.scatter(x, y, alpha=0.6)
coef = np.polyfit(x, y, 1)
xp = np.linspace(x.min(), x.max(), 100)
plt.plot(xp, coef[0]*xp + coef[1])
plt.title("Relacion: Uso FinTech vs Inclusion financiera (indice)")
plt.xlabel("VI_uso_tecnologias")
plt.ylabel("VD_inclusion_financiera")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scatter_uso_vs_inclusion.png", dpi=200)
plt.close()

# 9.3 Heatmap correlaciones Spearman
plt.figure()
plt.imshow(corr_spearman.values)
plt.xticks(range(len(index_cols)), index_cols, rotation=45, ha="right")
plt.yticks(range(len(index_cols)), index_cols)
plt.title("Matriz de correlacion (Spearman) entre indices")
plt.colorbar()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "heatmap_correlaciones.png", dpi=200)
plt.close()

# ----------------------------
# 10) Guardar Excel de salida (varias hojas)
# ----------------------------
xlsx_out = OUT_DIR / "analisis_fintech_milagro.xlsx"

with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="data_con_indices", index=False)
    alpha_summary.to_excel(writer, sheet_name="alpha", index=False)
    descriptivos.to_excel(writer, sheet_name="descriptivos")
    corr_spearman.to_excel(writer, sheet_name="correlaciones_spearman")
    reg_info.to_excel(writer, sheet_name="regresion_info", index=False)
    if not reg_coef.empty:
        reg_coef.to_excel(writer, sheet_name="regresion_coef", index=False)

# ----------------------------
# 11) Mensajes finales
# ----------------------------
print("\n=== SALIDAS GENERADAS ===")
print("Excel:", xlsx_out)
print("Carpeta graficas:", PLOTS_DIR)
print("\n=== ALFA DE CRONBACH ===")
print(alpha_summary)
print("\n=== CORRELACIONES (Spearman) ===")
print(corr_spearman)
if not reg_coef.empty:
    print("\n=== REGRESION (coeficientes) ===")
    print(reg_coef.head(10))
print("\nListo.")
