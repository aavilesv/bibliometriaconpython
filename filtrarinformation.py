import pandas as pd

# ===============================
# 1. CARGAR DATASET FINAL (426)
# ===============================
df = pd.read_csv(r"G:\Mi unidad\2026\Master Elka Almeida\datawos_scopus.csv")

print("Registros cargados:", len(df))

# ===============================
# 2. UNIFICAR CAMPOS DE TEXTO
# ===============================
df["text_combined"] = (
    df["Title"].fillna("").astype(str) + " " +
    df["Abstract"].fillna("").astype(str) + " " +
    df["bothkeywords"].fillna("").astype(str)
).str.lower()

# ===============================
# 3. DEFINICIÓN DE BLOQUES PCC
# ===============================

# Contexto: Educación Superior
he_terms = [
      "higher education", "tertiary education",
    "university", "universities",
]

# Fenómeno: Desarrollo Profesional Docente
dpd_terms = [
  "faculty development",
    "professional development",
    "academic staff development"
]

# Contexto Digital
digital_terms = [
    "digital transformation",
    "digitalization",
    "educational technology",
    "artificial intelligence",
    "generative ai",
    "technology integration"
]

# ===============================
# 4. FUNCIÓN DE COINCIDENCIA
# ===============================
def contains_any(text, term_list):
    return any(term in text for term in term_list)

# ===============================
# 5. CREACIÓN DE FLAGS
# ===============================
df["HE_flag"] = df["text_combined"].apply(lambda x: contains_any(x, he_terms))
df["DPD_flag"] = df["text_combined"].apply(lambda x: contains_any(x, dpd_terms))
df["DIGITAL_flag"] = df["text_combined"].apply(lambda x: contains_any(x, digital_terms))

# ===============================
# 6. CLASIFICACIÓN DE CANDIDATOS
# ===============================
df["PreScreen_Candidate"] = (
    df["HE_flag"] &
    df["DPD_flag"] &
    df["DIGITAL_flag"]
)

# ===============================
# 7. RESUMEN PARA PRISMA
# ===============================
total_records = len(df)
candidate_records = df["PreScreen_Candidate"].sum()

print("Total registros:", total_records)
print("Candidatos tras pre-screening automático:", candidate_records)

# ===============================
# 8. EXPORTAR RESULTADO
# ===============================
df.to_csv(r"G:\Mi unidad\2026\Master Elka Almeida\dataset_prescreened.csv", index=False)

print("Archivo generado: dataset_prescreened.csv")