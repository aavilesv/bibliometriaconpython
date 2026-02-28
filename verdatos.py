import pandas as pd
import numpy as np

# =====================================================
# Configuración de visualización (NO ocultar nada)
# =====================================================
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# =====================================================
# Cargar datasets
# =====================================================
df1 = pd.read_csv(
    r"G:\Mi unidad\2026\Master Rosmery Montiel\dataset1_numerico_final.csv"
)

df2 = pd.read_csv(
    r"G:\Mi unidad\2026\Master Rosmery Montiel\dataset2_numerico_final.csv"
)

# =====================================================
# 1. Ver nombres EXACTOS de columnas con índice
# =====================================================
print("\n========== COLUMNAS DATASET 1 ==========")
for i, col in enumerate(df1.columns):
    print(f"{i:02d} → {col}")

print("\n========== COLUMNAS DATASET 2 ==========")
for i, col in enumerate(df2.columns):
    print(f"{i:02d} → {col}")

# =====================================================
# 2. Resumen de número de valores únicos por columna
# =====================================================
resumen_unicos_df1 = pd.DataFrame({
    "columna": df1.columns,
    "n_unicos": [df1[col].nunique(dropna=False) for col in df1.columns]
}).sort_values("n_unicos", ascending=False)

resumen_unicos_df2 = pd.DataFrame({
    "columna": df2.columns,
    "n_unicos": [df2[col].nunique(dropna=False) for col in df2.columns]
}).sort_values("n_unicos", ascending=False)

print("\n========== RESUMEN ÚNICOS DATASET 1 ==========")
print(resumen_unicos_df1)

print("\n========== RESUMEN ÚNICOS DATASET 2 ==========")
print(resumen_unicos_df2)

# =====================================================
# 3. Ver valores ÚNICOS REALES (arreglo) por columna
# =====================================================
print("\n========== VALORES ÚNICOS (DATASET 1) ==========")
for col in df1.columns:
    print("\n------------------------------------")
    print(f"COLUMNA: {col}")
    print(df1[col].unique())

print("\n========== VALORES ÚNICOS (DATASET 2) ==========")
for col in df2.columns:
    print("\n------------------------------------")
    print(f"COLUMNA: {col}")
    print(df2[col].unique())

