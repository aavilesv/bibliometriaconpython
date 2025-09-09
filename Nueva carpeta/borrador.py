import pandas as pd

# === Configura aquí tus archivos/columnas ===
INPUT_XLSX  = r"G:\Mi unidad\2025\codigos bibliometria NPL\Nueva carpeta\data_.xlsx"                 # tu archivo de entrada
OUTPUT_XLSX = "articulos_exploded.xlsx"        # archivo de salida
COL_UNIS    = "Combined_universities"          # columna con las universidades

# 1) Leer
df = pd.read_excel(INPUT_XLSX, dtype=str)

if COL_UNIS not in df.columns:
    raise ValueError(f"No existe la columna '{COL_UNIS}' en el Excel.")

# 2) Asegurar texto y manejar vacíos
df[COL_UNIS] = df[COL_UNIS].fillna("").astype(str)

# 3) Dividir por ';' (ignorando espacios alrededor) y EXPLODE
#    - Cada valor separado por ';' se convierte en una fila nueva.
split_series = df[COL_UNIS].str.split(r"\s*;\s*")
exploded = (
    df.drop(columns=[COL_UNIS])
      .join(split_series.explode().rename("University"))
)

# 4) Limpiar espacios y eliminar filas vacías
exploded["University"] = exploded["University"].str.strip()
exploded = exploded[exploded["University"].astype(bool)]

# (Opcional) Normalizar mínimamente para reducir duplicados por espacios raros
exploded["University_clean"] = (
    exploded["University"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 5) Conteo de artículos por universidad (según nombre normalizado)
counts = (
    exploded["University_clean"]
    .value_counts(dropna=False)
    .rename_axis("University_clean")
    .reset_index(name="n_articulos")
)

# 6) Guardar a Excel con dos hojas:
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
    exploded.to_excel(w, index=False, sheet_name="exploded")   # filas explotadas
    counts.to_excel(w,   index=False, sheet_name="counts")     # conteo por uni

print("Hecho ✅  Archivo generado:", OUTPUT_XLSX)
