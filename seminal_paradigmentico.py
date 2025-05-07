
import pandas as pd, numpy as np
import re
#df = pd.read_csv("G:\\Mi unidad\\2025\\Master Almeida Monge Elka Jennifer\\data\\datascopus.csv")  
RUTA = r"G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv"

UMBRAL_CITAS = 15                # ⇦ cámbialo a 25, 30, 40… y se recalcula todo
CUTOFF_YEAR = 2024            # edad al cierre de la búsqueda

# -----------------------------------------------------------------------
# 1) Carga y limpieza básica
df = pd.read_csv(RUTA)
df["Cited by"] = df["Cited by"].fillna(0).astype(int)
df["Year"]     = df["Year"].fillna(df["Year"].mode()[0]).astype(int)

#df = df[df["Document Type"].str.lower() == "article"]
# 2) Renombre de columnas útiles
df = df.rename(columns={
    "Cited by":  "Citas",
    "Abstract":  "Resumen",
    "Document Type": "TipoDoc"
})
# ── 3. Antigüedad y orden por citas ──────────────────────────────────
df["Antiguedad"] = CUTOFF_YEAR - df["Year"]
df = df.sort_values(["Citas","Year"], ascending=[False,True])

## unir las columnas
# 2) texto unificado (title + abstract + keywords)
text_cols = ["Title", "Resumen", "Author Keywords", "Index Keywords"]
df["FullText"] = df[text_cols].fillna("").agg(" ".join, axis=1)


# ── 4. Heurística empírica estricta (≥2 señales) ─────────────────────
pattern_method = (
   r"\b("
    r"experiment|randomized|controlled trial|quasi[- ]?experimental|"
    r"pre[- ]?test|post[- ]?test|survey|questionnaire|interview|"
    r"mixed(\s*methods)?|quantitative|qualitative|"
    r"case[- ]?study|regression|anova"
    r")\b"
)
pattern_data = r"(data (were|was) collected|sample size|participants|sample|n\s*=|n =\s*\d)"
    

df["IsExperimental"] = (
    df["FullText"].str.contains(pattern_method, flags=re.I, regex=True) &
    df["FullText"].str.contains(pattern_data,  flags=re.I, regex=True)
)

# 4) Clasificación final = experimental ∧ citas>umbral
df["Clasificacion"] = np.where(
    df["IsExperimental"] & (df["Citas"] > UMBRAL_CITAS),
    "Paradigmático", "Seminal"
)

# 5) Resumen rápido
print(f"Total artículos……………… {len(df)}")
print(f"Paradigmáticos……………… {sum(df['Clasificacion']=='Paradigmático')}")
print(f"Seminales…………………. {sum(df['Clasificacion']=='Seminal')}")

# 6) Tabla de distribución de citas (P0–P100)
print("\nPercentiles de citas (todo el set):")
print(df["Citas"].describe(percentiles=[.1,.25,.5,.75,.9,.95]))

# 7) Tira de cortes para que compares (10 – 40 citas)
for thr in range(10, 45, 5):
    n_par = sum(df["IsExperimental"] & (df["Citas"] > thr))
    print(f"  > {thr:2d} citas  →  {n_par:4d} paradigmáticos")

# 8) Guarda en Excel
with pd.ExcelWriter("clasificacion_review.xlsx") as w:
    df.to_excel(w, sheet_name="Todos", index=False)
    df[df["Clasificacion"]=="Paradigmático"].to_excel(w, sheet_name="Paradigmáticos", index=False)
    df[df["Clasificacion"]=="Seminal"].to_excel(w, sheet_name="Seminales", index=False)
print("\nExcel generado: clasificacion_review.xlsx")
