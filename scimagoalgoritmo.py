# =====================================================
# MODELO DE CLASIFICACIÓN DE RIESGO EDITORIAL
# Scimago longitudinal + Scopus mensual
# =====================================================

import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, process

# =====================================================
# 1. RUTAS
# =====================================================
RUTA = r"G:\Mi unidad\scimago"

SCIMAGO_FILE = RUTA + r"\scimago_2013_2024_longitudinal.csv"
SCOPUS_FILE  = RUTA + r"\datarevistasactivas.xlsx"
OUTPUT_FILE  = RUTA + r"\clasificacion_riesgo_editorial.xlsx"

# =====================================================
# 2. FUNCIONES DE NORMALIZACIÓN
# =====================================================
def normalizar_issn(x):
    if pd.isna(x):
        return ""
    return re.sub(r'[^0-9X]', '', str(x).upper())

def normalizar_titulo(x):
    if pd.isna(x):
        return ""
    x = x.lower()
    x = re.sub(r'[^a-z0-9 ]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

# =====================================================
# 3. CARGA DE DATOS
# =====================================================
df_scimago = pd.read_csv(SCIMAGO_FILE)
df_scopus  = pd.read_excel(SCOPUS_FILE)

# =====================================================
# 4. NORMALIZACIÓN DE VARIABLES NUMÉRICAS (CRÍTICO)
# =====================================================
cols_numericas = [
    "SJR",
    "Total Docs.",
    "Citations / Doc. (2years)"
]

for col in cols_numericas:
    df_scimago[col] = (
        df_scimago[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("nan", np.nan)
    )
    df_scimago[col] = pd.to_numeric(df_scimago[col], errors="coerce")

# =====================================================
# 5. NORMALIZACIÓN BÁSICA
# =====================================================
df_scimago["TITLE_N"] = df_scimago["Title"].apply(normalizar_titulo)

df_scopus["TITLE_N"]  = df_scopus["Source Title"].apply(normalizar_titulo)
df_scopus["ISSN_N"]   = df_scopus["ISSN"].apply(normalizar_issn)
df_scopus["EISSN_N"]  = df_scopus["EISSN"].apply(normalizar_issn)

scopus_titles = df_scopus["TITLE_N"].dropna().tolist()

# =====================================================
# 6. MATCHING SCOPUS (ISSN → TÍTULO)
# =====================================================
def obtener_estado_scopus(row, threshold=90):
    issn_val  = row.get("Issn", "")
    title_val = row.get("Title", "")

    # ---- MATCH POR ISSN ----
    issns = [
        normalizar_issn(i)
        for i in str(issn_val).replace(";", ",").split(",")
        if normalizar_issn(i)
    ]

    for issn in issns:
        match = df_scopus[
            (df_scopus["ISSN_N"] == issn) |
            (df_scopus["EISSN_N"] == issn)
        ]
        if not match.empty:
            fila = match.iloc[0]
            return fila["Active or Inactive"], fila["Coverage"], "ISSN"

    # ---- MATCH POR TÍTULO ----
    title_n = normalizar_titulo(title_val)
    if title_n:
        result = process.extractOne(
            title_n,
            scopus_titles,
            scorer=fuzz.WRatio
        )
        if result:
            _, score, idx = result
            if score >= threshold:
                fila = df_scopus.iloc[idx]
                return fila["Active or Inactive"], fila["Coverage"], "TITLE"

    return "Not in Scopus", None, "NONE"

# =====================================================
# 7. APLICAR MATCHING SCOPUS
# =====================================================
df_scimago[
    ["scopus_status", "scopus_coverage", "scopus_match_method"]
] = df_scimago.apply(
    lambda r: pd.Series(obtener_estado_scopus(r)),
    axis=1
)

# =====================================================
# 8. EVENTO EDITORIAL (NO TEMPORAL)
# =====================================================
df_scimago["discontinued_flag"] = (df_scimago["scopus_status"] == "Inactive").astype(int)

df_scimago["discontinued_year"] = (
    df_scimago["scopus_coverage"]
    .astype(str)
    .str.extract(r'(\d{4})$')
    .astype(float)
)

# =====================================================
# 9. MÉTRICAS LONGITUDINALES (SCIMAGO)
# =====================================================
df_scimago = df_scimago.sort_values(["TITLE_N", "year"])

# ---- Variación abrupta de SJR ----
df_scimago["sjr_var"] = (
    df_scimago
    .groupby("TITLE_N")["SJR"]
    .pct_change(fill_method=None)
    .abs()
)

# ---- Crecimiento anómalo de documentos ----
df_scimago["docs_ma3"] = (
    df_scimago.groupby("TITLE_N")["Total Docs."]
    .transform(lambda x: x.rolling(3, min_periods=2).mean())
)

df_scimago["docs_growth"] = df_scimago["Total Docs."] / df_scimago["docs_ma3"]

# ---- Persistencia de citas bajas ----
df_scimago["low_cite_flag"] = (df_scimago["Citations / Doc. (2years)"] < 0.3).astype(int)

df_scimago["low_cite_persistent"] = (
    df_scimago.groupby("TITLE_N")["low_cite_flag"]
    .transform(lambda x: x.rolling(3, min_periods=3).sum() >= 3)
    .astype(int)
)

# =====================================================
# 10. SISTEMA DE PUNTUACIÓN (HEURÍSTICO)
# =====================================================
df_scimago["risk_score"] = 0

df_scimago.loc[df_scimago["sjr_var"] > 0.8, "risk_score"] += 2
df_scimago.loc[df_scimago["docs_growth"] > 3, "risk_score"] += 2
df_scimago.loc[df_scimago["low_cite_persistent"] == 1, "risk_score"] += 2
df_scimago.loc[df_scimago["discontinued_flag"] == 1, "risk_score"] += 3

# =====================================================
# 11. CLASIFICACIÓN FINAL
# =====================================================
def clasificar(score):
    if score >= 6:
        return "Alto riesgo editorial"
    elif score >= 3:
        return "Riesgo editorial medio"
    else:
        return "Bajo riesgo editorial"

df_scimago["risk_class"] = df_scimago["risk_score"].apply(clasificar)

# =====================================================
# 12. RESULTADO A NIVEL REVISTA
# =====================================================
resultado_revistas = (
    df_scimago.groupby("TITLE_N")
    .agg({
        "risk_score": "max",
        "risk_class": "last",
        "discontinued_flag": "max",
        "discontinued_year": "max",
        "scopus_match_method": "last"
    })
    .reset_index()
)

# =====================================================
# 13. GUARDAR RESULTADOS
# =====================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    df_scimago.to_excel(writer, sheet_name="Detalle_Longitudinal", index=False)
    resultado_revistas.to_excel(writer, sheet_name="Clasificacion_Revistas", index=False)

print("Proceso completado correctamente.")
print(f"Archivo generado en: {OUTPUT_FILE}")

print("\nDistribución de clases:")
print(resultado_revistas["risk_class"].value_counts())
