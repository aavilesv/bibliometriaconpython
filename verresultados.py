import pandas as pd
import os

# =====================================================
# RUTAS
# =====================================================
base_path = r"G:\Mi unidad\2026\Master Rosmery Montiel"

df1_path = os.path.join(base_path, "dataset1_numerico_final.csv")
df2_path = os.path.join(base_path, "dataset2_numerico_final.csv")

out_excel = os.path.join(base_path, "Resultados_Analisis_NE2A.xlsx")

# =====================================================
# CARGA
# =====================================================
df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)

# =====================================================
# ================= DATASET 1 =========================
# =====================================================

# --- Descriptivos
desc_d1 = df1.describe().round(3)

# --- Frecuencias (solo contexto)
freq_d1 = {
    "genero": df1["genero"].value_counts(),
    "q1_nivel": df1["q1"].value_counts(),
    "q2_experiencia": df1["q2"].value_counts(),
    "q3_formacion": df1["q3"].value_counts(),
    "q4_necesidades": df1["q4"].value_counts(),
    "q5_recursos_apoyo": df1["q5"].value_counts(),
    "q17_discapacidad": df1["q17"].value_counts(),
}

freq_d1_df = pd.concat(freq_d1, axis=1).fillna(0).astype(int)

# --- Correlaciones relevantes
corr_d1 = df1[
    ["q2","q3","q4","q5","q6","q8","q12","q15","q17"]
].corr().round(3)

# =====================================================
# ================= DATASET 2 =========================
# =====================================================

# --- Descriptivos
desc_d2 = df2.describe().round(3)

# --- Frecuencias (solo contexto)
freq_d2 = {
    "genero": df2["genero"].value_counts(),
    "q2_experiencia": df2["q2"].value_counts(),
    "q3_conocimiento": df2["q3"].value_counts(),
    "q4_claridad": df2["q4"].value_counts(),
    "q6_acomparecibido": df2["q6"].value_counts()
}

freq_d2_df = pd.concat(freq_d2, axis=1).fillna(0).astype(int)

# --- Correlaciones relevantes
corr_d2 = df2[
    ["q3","q4","q5","q6","q10","q11"]
].corr().round(3)

# =====================================================
# GUARDAR EXCEL FINAL
# =====================================================
with pd.ExcelWriter(out_excel, engine="xlsxwriter") as writer:
    desc_d1.to_excel(writer, sheet_name="D1_Descriptivos")
    freq_d1_df.to_excel(writer, sheet_name="D1_Frecuencias")
    corr_d1.to_excel(writer, sheet_name="D1_Correlaciones")

    desc_d2.to_excel(writer, sheet_name="D2_Descriptivos")
    freq_d2_df.to_excel(writer, sheet_name="D2_Frecuencias")
    corr_d2.to_excel(writer, sheet_name="D2_Correlaciones")

print("✔ Archivo Excel de resultados creado correctamente")
print("📁 Ubicación:", out_excel)
