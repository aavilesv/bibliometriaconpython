import pandas as pd
import numpy as np

# =====================================================
# RUTAS
# =====================================================
df1_path = r"G:\Mi unidad\2026\Master Rosmery Montiel\dataset1.xlsx"
df2_path = r"G:\Mi unidad\2026\Master Rosmery Montiel\dataset2.xlsx"

out_df1 = r"G:\Mi unidad\2026\Master Rosmery Montiel\dataset1_numerico_final.csv"
out_df2 = r"G:\Mi unidad\2026\Master Rosmery Montiel\dataset2_numerico_final.csv"

# =====================================================
# CARGA
# =====================================================
df1 = pd.read_excel(df1_path)
df2 = pd.read_excel(df2_path)

# =====================================================
# LIMPIEZA TEXTO (sin warnings)
# =====================================================
def clean_text(x):
    if isinstance(x, str):
        return " ".join(x.lower().strip().split())
    return x

df1 = df1.map(clean_text)
df2 = df2.map(clean_text)

# =====================================================
# DICCIONARIOS
# =====================================================
map_genero = {"femenino": 1, "masculino": 2}

map_experiencia = {
    "menos de 1 año": 1,
    "1-5 años": 2,
    "6-10 años": 3,
    "más de 10 años": 4
}

map_si_no = {
    "si": 1,
    "sí": 1,
    "no": 0,
    "no estoy seguro": 2,
    "no estoy segura": 2
}

map_frecuencia = {
    "raramente": 1,
    "a veces": 2,
    "frecuentemente": 3,
    "siempre": 4
}

map_likert_4 = {
    "totalmente de acuerdo": 4,
    "de acuerdo": 3,
    "parcialmente": 2,
    "en desacuerdo": 1
}

# =====================================================
# MULTIRESPUESTA → ESCALAR (ROBUSTA)
# =====================================================
def multiselect_to_score(x):
    if pd.isna(x):
        return np.nan

    # si ya es numérico, no tocar
    if isinstance(x, (int, float, np.integer, np.floating)):
        return x

    x = str(x).strip()
    if x == "":
        return np.nan

    n = len([i for i in x.split(",") if i.strip() != ""])
    if n <= 1:
        return 1
    elif n == 2:
        return 2
    else:
        return 3

# =====================================================
# ================= DATASET 1 =========================
# =====================================================
df1["genero"] = df1["genero"].map(map_genero)
df1["q1"] = df1["q1"].map({"bachillerato": 1, "educación basica": 2})
df1["q2"] = df1["q2"].map(map_experiencia)
df1["q3"] = df1["q3"].map(map_si_no)
df1["q4"] = df1["q4"].map(map_frecuencia)
df1["q7"] = df1["q7"].map(map_frecuencia)
df1["q9"] = df1["q9"].map(map_si_no).fillna(0)

# multirespuesta (LAS QUE FALTABAN INCLUIDAS)
for c in ["q5", "q6", "q8", "q10", "q19"]:
    if c in df1.columns:
        df1[c] = df1[c].apply(multiselect_to_score)

# escalas simples
df1["q11"] = df1["q11"].map({
    "evaluaciones formales (exámenes, pruebas)": 1,
    "feedback continuo del estudiante y familia": 2,
    "observación del comportamiento y la participación": 3
})

df1["q12"] = df1["q12"].map({
    "no me afecta": 0,
    "me afecta mínimamente": 1,
    "me causa estrés moderado": 2
})

df1["q13"] = df1["q13"].map({
    "frustración": 1,
    "satisfacción por ayudar": 2,
    "empatía, satisfacción por ayudar": 3
})

df1["q14"] = df1["q14"].map({
    "no tengo una estrategia clara": 1,
    "consulto con especialistas": 2,
    "busco apoyo en colegas": 3,
    "practico técnicas de autocuidado (meditación, ejercicio, etc.)": 4
})

# eliminar abiertas y basura
df1.drop(columns=["q16", "q18", "q19.1"], inplace=True, errors="ignore")

# =====================================================
# ================= DATASET 2 =========================
# =====================================================
df2["genero"] = df2["genero"].map(map_genero)
df2["q1"] = 1
df2["q2"] = df2["q2"].map(map_experiencia)

df2["q3"] = df2["q3"].map({
    "sí, las conozco bien y sé cómo se aplican": 3,
    "he escuchado de ellas, pero no las comprendo": 1
})

df2["q4"] = df2["q4"].map(map_likert_4)
df2["q5"] = df2["q5"].map({"muy preparado/a": 3, "medianamente preparado/a": 2})
df2["q6"] = df2["q6"].map({"sí, en todas las fases": 3, "solo en algunas fases": 2})

# constantes
for c in ["q7","q8","q9","q12","q14","q15","q16"]:
    if c in df2.columns:
        df2[c] = 3

df2["q10"] = df2["q10"].map({"totalmente de acuerdo": 4, "de acuerdo": 3})
df2["q11"] = df2["q11"].map({"sí, en gran medida": 3, "parcialmente": 2})
df2["q13"] = df2["q13"].map({"sí, completamente": 3, "parcialmente": 2})

# multirespuesta
for c in ["q21", "q22"]:
    if c in df2.columns:
        df2[c] = df2[c].apply(multiselect_to_score)

# q17–q20 = 'sí'
for c in ["q17", "q18", "q19", "q20"]:
    if c in df2.columns:
        df2[c] = df2[c].astype(str).str.lower().map({"sí": 1, "si": 1}).fillna(0).astype(int)

# =====================================================
# VALIDACIÓN FINAL
# =====================================================
print("OBJETOS DF1:", df1.select_dtypes(include="object").columns.tolist())
print("OBJETOS DF2:", df2.select_dtypes(include="object").columns.tolist())

# =====================================================
# GUARDAR
# =====================================================
df1.to_csv(out_df1, index=False)
df2.to_csv(out_df2, index=False)

print("✔ Normalización completa, sin columnas nuevas y sin errores")
