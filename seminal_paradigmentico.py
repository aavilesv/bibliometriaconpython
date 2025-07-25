"""
clasificacion_pipeline.py
-------------------------
• Limpia CSV (WoS + Scopus)
• Marca duplicados
• Criba por experimentalidad y citas
• Genera archivos por fase para PRISMA
• Crea subconjuntos: Seminales_75-90, Regionales y Union_final
"""

import sys, re, pandas as pd, numpy as np
from rapidfuzz import process, fuzz            # para coincidencias de país

# ─────────────── 0 · Parámetros ──────────────────────────
RUTA_FUENTE = sys.argv[1] if len(sys.argv) > 1 \
              else r"G:\\Mi unidad\\2025\\Master Italo Palacios\\articulo\\datawos_scopus.csv"

UMBRAL_CITAS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CUTOFF_YEAR  = 2025
OUT_DIR      = r"G:\\Mi unidad\\2025\\Master Italo Palacios\\articulo"

# Países considerados “región de interés”
REGIONAL_LIST = [
    "Ecuador","Peru","Colombia","Bolivia","Chile","Argentina","Brazil","Mexico",
    "Costa Rica","Panama","Guatemala","El Salvador","Honduras","Nicaragua","Paraguay",
    "Uruguay","Venezuela","Cuba","Dominican Republic",
    # África
    "South Africa","Kenya","Uganda","Nigeria","Ghana","Ethiopia","Tanzania",
    # Asia ingreso medio-bajo
    "India","Indonesia","Philippines","Vietnam","Bangladesh","Pakistan","Sri Lanka"
]
REGIONAL_SET = {c.lower() for c in REGIONAL_LIST}

print(f"▶ Fuente: {RUTA_FUENTE}\n▶ Umbral citas: {UMBRAL_CITAS}\n▶ Carpeta salida: {OUT_DIR}\n")

# ─────────────── 1 · Carga y deduplicación ─────────────────────
df0 = pd.read_csv(RUTA_FUENTE)
df0["dup_key"] = np.where(df0["DOI"].notna(),
                          df0["DOI"].str.lower(),
                          df0["Title"].str.lower().str.slice(0,120) + "_" + df0["Year"].astype(str))
df0["IsDuplicate"] = df0.duplicated("dup_key", keep="first")
df  = df0.loc[~df0["IsDuplicate"]].copy()

# ─────────────── 2 · Limpieza básica ───────────────────────────
df["Cited by"] = df["Cited by"].fillna(0).astype(int)
df["Year"]     = df["Year"].fillna(df["Year"].mode()[0]).astype(int)
df = df.rename(columns={"Cited by":"Citas","Abstract":"Resumen","Document Type":"TipoDoc"})
text_cols      = ["Title","Resumen","Author Keywords","Index Keywords"]
df["FullText"] = df[text_cols].fillna("").agg(" ".join, axis=1)

# ─────────────── 3 · Cribado de experimentalidad ───────────────
pattern_method = (
 r"\b(experiment|empirical|randomized|controlled trial|quasi[- ]?experimental|"
 r"pre[- ]?test|post[- ]?test|survey|questionnaire|interview|"
 r"mixed(\s*methods)?|quantitative|qualitative|case[- ]?study|regression|anova)\b"
)
pattern_data = r"(data (were|was) collected|sample size|participants|sample|n\s*=|n =\s*\d)"

df["IsExperimental"] = (
    df["FullText"].str.contains(pattern_method, flags=re.I, regex=True) &
    df["FullText"].str.contains(pattern_data,   flags=re.I, regex=True)
)

# ─────────────── 4 · Clasificación por citas ───────────────────
df["Clasificacion"] = np.select(
    [df["IsExperimental"] & (df["Citas"] > UMBRAL_CITAS)],
    ["Paradigmático"],
    default="Seminal"
)

# ─────────────── 5 · Subconjunto Seminales 75-90 ───────────────
q75 = df.loc[df["Clasificacion"]=="Seminal","Citas"].quantile(0.75)
q90 = df.loc[df["Clasificacion"]=="Seminal","Citas"].quantile(0.90)

seminal_75_90 = df[(df["Clasificacion"]=="Seminal") &
                   (df["Citas"]>=q75) & (df["Citas"]<=q90)].copy()

# ─────────────── 6 · Subconjunto Regional ──────────────────────
def is_regional(country_field:str) -> bool:
    if not isinstance(country_field,str): return False
    best, score, _ = process.extractOne(country_field.lower(), REGIONAL_SET,
                                        scorer=fuzz.partial_ratio)
    return score >= 95     # umbral laxo para variaciones ortográficas

text_colsseminales      = ["Authors with affiliations","Affiliations","Author Keywords","Index Keywords"]
df["Affiliationsfull"] = df[text_cols].fillna("").agg(" ".join, axis=1)
df["RegionalFlag"] = df["Affiliationsfull"].apply(is_regional)
regional_subset    = df[df["RegionalFlag"]].copy()

# ─────────────── 7 · Unión final y guardado ────────────────────
union_final = pd.concat([seminal_75_90, regional_subset]).drop_duplicates("dup_key")

with pd.ExcelWriter(f"{OUT_DIR}\\03_clasificacion_review.xlsx",
                    mode="a", engine="openpyxl", if_sheet_exists="replace") as w:
    seminal_75_90.to_excel(w, sheet_name="Seminales_75-90", index=False)
    regional_subset.to_excel(w, sheet_name="Regionales",       index=False)
    union_final.to_excel(w,    sheet_name="Union_final",       index=False)

print("\n✔ Hojas adicionales creadas:")
print("   • Seminales_75-90")
print("   • Regionales")
print("   • Union_final")
