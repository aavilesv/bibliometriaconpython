"""
clasificacion_pipeline.py
-------------------------
• Fusiona y limpia CSV (WoS + Scopus)
• Marca duplicados (DOI o Título+Año)
• Detecta señales metodológicas (ES/EN) para experimentalidad
• Permite umbral por citas: métrica ("brutas" | "por_anio") y tipo ("abs" | "pct")
• Clasifica solo en: Paradigmáticos y Seminales
• Exporta SOLO esas dos hojas (sobrescribe el Excel)
• Imprime conteos PRISMA en consola
"""

import sys, re
from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────── 0 · Parámetros base ──────────────────────────
RUTA_FUENTE = sys.argv[1] if len(sys.argv) > 1 \
    else r"G:\Mi unidad\Master en administración y empresas\articulo 3\data\datawos_scopus.csv"

UMBRAL_CITAS_DEF = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # respaldo si no se pasa VALOR_UMBRAL
CUTOFF_YEAR  = 2025
OUT_DIR      = Path(r"G:\Mi unidad\Master en administración y empresas\articulo 3\data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
XLSX_PATH    = OUT_DIR / "03_clasificacion_review.xlsx"            # se sobrescribe

# ─────────────── 0bis · Parámetros de umbral de citas ─────────
# argv[3]=METRICA_CITAS  ("brutas" | "por_anio")
# argv[4]=TIPO_UMBRAL    ("abs" | "pct")
# argv[5]=VALOR_UMBRAL   (num: si "abs" → valor; si "pct" → 0–1 ó 0–100)
# argv[6]=PCT_SEMINAL    (num: 0–1 ó 0–100; percentil para "muy citado" en Seminal; default 0.90)
METRICA_CITAS = (sys.argv[3] if len(sys.argv) > 3 else "brutas").lower()
TIPO_UMBRAL   = (sys.argv[4] if len(sys.argv) > 4 else "abs").lower()
VALOR_UMBRAL  = float(sys.argv[5]) if len(sys.argv) > 5 else float(UMBRAL_CITAS_DEF)
PCT_SEMINAL   = float(sys.argv[6]) if len(sys.argv) > 6 else 0.90

if PCT_SEMINAL > 1:
    PCT_SEMINAL /= 100.0
if TIPO_UMBRAL == "pct" and VALOR_UMBRAL > 1:
    VALOR_UMBRAL /= 100.0

print(f"▶ Fuente: {RUTA_FUENTE}")
print(f"▶ Métrica de citas: {METRICA_CITAS} | Tipo umbral: {TIPO_UMBRAL} | Valor: {VALOR_UMBRAL}")
print(f"▶ Percentil 'muy citado' para Seminal: {PCT_SEMINAL:.2f}")
print(f"▶ Carpeta salida: {OUT_DIR}\n")

# ─────────────── 1 · Carga y deduplicación ─────────────────────
df0 = pd.read_csv(RUTA_FUENTE)

# Normalización de DOI, Título y Año
doi_str   = df0.get("DOI", pd.Series(index=df0.index, dtype="object")).fillna("").astype(str).str.strip().str.lower()
title_str = df0.get("Title", pd.Series(index=df0.index, dtype="object")).fillna("").astype(str).str.strip().str.lower()
year_col  = pd.to_numeric(df0.get("Year", pd.Series(index=df0.index)), errors="coerce")
year_str  = year_col.astype("Int64").astype(str)

df0["dup_key"] = np.where(doi_str != "", doi_str, title_str.str.slice(0,120) + "_" + year_str)
df0["IsDuplicate"] = df0.duplicated("dup_key", keep="first")
df  = df0.loc[~df0["IsDuplicate"]].copy()

# ─────────────── 2 · Limpieza básica ───────────────────────────
df["Citas"]   = pd.to_numeric(df.get("Cited by", 0), errors="coerce").fillna(0).astype(int)
# Resumen puede venir como "Abstract" o "Resumen"
if "Abstract" in df.columns:
    df["Resumen"] = df["Abstract"].fillna("")
else:
    df["Resumen"] = df.get("Resumen", "").fillna("")
df["TipoDoc"] = df.get("Document Type", "")

if "Year" in df.columns:
    if df["Year"].isna().all():
        df["Year"] = CUTOFF_YEAR
    else:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(df["Year"].mode().iloc[0]).astype(int)
else:
    df["Year"] = CUTOFF_YEAR

# Texto para evaluación metodológica
text_cols = [c for c in ["Title","Resumen","Author Keywords","Index Keywords"] if c in df.columns]
df["FullText"] = df[text_cols].fillna("").agg(" ".join, axis=1) if text_cols else ""

# ─────────────── 3 · Patrones enriquecidos (ES/EN) ─────────────
def rx(p): 
    return re.compile(p, flags=re.I)

# Experimental / quasi-experimental / causal design
RX_DESIGN = rx(
    r"\b("
    r"randomi[sz]ed(?:\s+controlled)?\s+trial|RCT|"
    r"quasi[-\s]?experimental|"
    r"field\s+experiment|laborator(y|ies)\s+experiment|laboratory\s+experiment|"
    r"difference[-\s]?in[-\s]?differences|diff[-\s]?in[-\s]?diff|DiD|"
    r"regression\s+discontinuity|RDD|"
    r"propensity\s+score(?:\s+matching)?|PSM|"
    r"instrumental\s+variables?|IV|"
    r"natural\s+experiment|"
    r"controlled\s+trial|trial|"
    r"A/?B\s*test"
    r")\b"
)

# Data / sample
RX_DATA = rx(
    r"(?:data\s+(?:were|was)\s+collected|"
    r"sample\s+size|n\s*=\s*\d+|sample\b|participants?\b|"
    r"survey|questionnaire|interview)"
)

# Statistics / inference
RX_STATS = rx(
    r"\b(t[-\s]?test|anova|manova|ancova|chi[-\s]?square|"
    r"regression|logit|probit|GLM|panel\s+data|time\s+series|"
    r"hazard|cox|kaplan[-\s]?meier|survival|"
    r"SEM|PLS|structural\s+equation\s+model(?:ing)?|SmartPLS|AMOS|LISREL|lavaan|CFA|EFA|"
    r"cronbach(?:'s)?\s*alpha|KMO|Bartlett|"
    r"CI\s*\(?\d{1,2}%\)?|confidence\s+interval|"
    r"p\s*[<≤]\s*0\.\d+)"
)

# Instrument / validity
RX_INSTR = rx(
    r"\b(Likert|instrument|validity|reliability|AVE|CR|HTMT|"
    r"convergent\s+validity|discriminant\s+validity)\b"
)

# Ethics
RX_ETHICS = rx(
    r"\b(IRB|ethical\s+approval|informed\s+consent)\b"
)

# Reviews / theory / bibliometrics (for Seminal)
RX_REVIEW_THEORY = rx(
    r"\b(systematic\s+review|meta[-\s]?analysis|scoping\s+review|umbrella\s+review|"
    r"literature\s+review|state\s+of\s+the\s+art|"
    r"mapping\s+review|bibliometric|scientometric|VOSviewer|Bibliometrix|Biblioshiny|CiteSpace|"
    r"conceptual\s+model|framework|theoretical\s+framework|"
    r"position\s+paper|viewpoint|perspective|opinion|essay)\b"
)
def contains(rx, s):
    if not isinstance(s, str) or not s:
        return False
    return bool(rx.search(s))

# Señales booleanas
sig_design  = df["FullText"].apply(lambda s: contains(RX_DESIGN, s))
sig_data    = df["FullText"].apply(lambda s: contains(RX_DATA, s))
sig_stats   = df["FullText"].apply(lambda s: contains(RX_STATS, s))
sig_instr   = df["FullText"].apply(lambda s: contains(RX_INSTR, s))
sig_ethics  = df["FullText"].apply(lambda s: contains(RX_ETHICS, s))
sig_review  = df["FullText"].apply(lambda s: contains(RX_REVIEW_THEORY, s))

# Experimentalidad: diseño + (datos o estadística)
is_experimental = (sig_design & (sig_data | sig_stats))

# ─────────────── 4 · Métrica de citas y umbral ─────────────────
edad = (CUTOFF_YEAR - df["Year"] + 1).clip(lower=1)
df["CitasPorAnio"] = df["Citas"] / edad

if METRICA_CITAS == "por_anio":
    serie_metric = df["CitasPorAnio"]
    nombre_metric = "CitasPorAnio"
else:
    serie_metric = df["Citas"]
    nombre_metric = "Citas"

def cumple_umbral(serie, tipo, valor):
    if tipo == "abs":
        return serie >= valor
    elif tipo == "pct":
        umbral = serie.quantile(valor)
        return serie >= umbral
    # fallback: absoluto
    return serie >= valor

# ─────────────── 5 · Clasificación ─────────────────────────────
df["IsParadigmatico"] = is_experimental & cumple_umbral(serie_metric, TIPO_UMBRAL, VALOR_UMBRAL)

# Percentil para "muy citado" en Seminal (se usa la misma métrica elegida)
q_seminal = serie_metric.quantile(PCT_SEMINAL) if serie_metric.notna().any() else 0

df["IsSeminal"] = (
    (~df["IsParadigmatico"]) & (
        sig_review | ((~is_experimental) & (serie_metric >= q_seminal))
    )
)

df["Clasificacion"] = np.select(
    [df["IsParadigmatico"], df["IsSeminal"]],
    ["Paradigmático", "Seminal"],
    default="Excluido"
)

# ─────────────── 6 · Exportación (solo 2 hojas) ────────────────
paradigmaticos = df[df["Clasificacion"].eq("Paradigmático")].copy()
seminales      = df[df["Clasificacion"].eq("Seminal")].copy()

# Ordenar por la métrica usada (descendente)
paradigmaticos = paradigmaticos.sort_values(serie_metric.name, ascending=False)
seminales      = seminales.sort_values(serie_metric.name,      ascending=False)

try:
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl", mode="w") as w:
        paradigmaticos.to_excel(w, sheet_name="Paradigmaticos", index=False)
        seminales.to_excel(w,      sheet_name="Seminales",      index=False)

    print("\n✔ Archivo escrito con 2 hojas:")
    print("   • Paradigmaticos")
    print("   • Seminales")
    print(f"📄 Archivo: {XLSX_PATH}")
except PermissionError:
    print(f"\n✖ No se pudo escribir en {XLSX_PATH}. Cierre el libro en Excel y ejecute de nuevo.")

# ─────────────── 7 · Conteos PRISMA en consola ────────────────
prisma = {
    "registros_totales": int(len(df0)),
    "duplicados_eliminados": int(df0["IsDuplicate"].sum()),
    "tras_deduplicar": int(len(df)),
    "incluidos_paradigmaticos": int(df["IsParadigmatico"].sum()),
    "incluidos_seminales": int(df["IsSeminal"].sum()),
    "excluidos": int((df["Clasificacion"]=="Excluido").sum())
}
print("\nPRISMA:", prisma)

# ─────────────── 8 · Resumen de configuración ──────────────────
print(f"\n→ Métrica usada: {nombre_metric}")
if TIPO_UMBRAL == "pct":
    print(f"→ Umbral Paradigmático (percentil): p={VALOR_UMBRAL:.2f} "
          f"(≈ {serie_metric.quantile(VALOR_UMBRAL):.3f} en {nombre_metric})")
else:
    print(f"→ Umbral Paradigmático (absoluto): ≥ {VALOR_UMBRAL} en {nombre_metric}")
print(f"→ Umbral Seminal (percentil): p={PCT_SEMINAL:.2f} "
      f"(≈ {q_seminal:.3f} en {nombre_metric})")

"""
Ejemplos:

1) Paradigmático con umbral absoluto 5 citas brutas; Seminal p90:
   python clasificacion_pipeline.py data.csv 2 brutas abs 5 0.90

2) Paradigmático con percentil 80 sobre citas por año; Seminal p95:
   python clasificacion_pipeline.py data.csv 2 por_anio pct 0.80 0.95
   (también acepta 80 y 95 en lugar de 0.80 y 0.95)
"""
