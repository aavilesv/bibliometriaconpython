import pandas as pd
import re

# Rutas
ruta_entrada = r"G:\Mi unidad\2025\codigos bibliometria NPL\databrazil.xlsx"
ruta_salida = r"G:\Mi unidad\2025\codigos bibliometria NPL\databrazil_unificado.xlsx"

# Leer Excel
df = pd.read_excel(ruta_entrada)

# Limpieza básica
for col in ["ISSN", "titulo", "Estrato", "area"]:
    df[col] = df[col].astype(str).str.strip()

# -------- NORMALIZACIÓN DEL TÍTULO --------
def normalizar_titulo(titulo):
    titulo = titulo.upper()
    titulo = re.sub(r"\(.*?\)", "", titulo)     # elimina (ONLINE), (IMPRESSO), etc.
    titulo = re.sub(r",\s*$", "", titulo)       # elimina comas finales
    titulo = re.sub(r"\s+", " ", titulo)        # normaliza espacios
    return titulo.strip()

df["titulo_norm"] = df["titulo"].apply(normalizar_titulo)

# Función para unir valores únicos con ";"
def unir_unicos(series):
    valores = series.dropna().unique()
    return "; ".join(sorted(valores))

# Agrupación final
df_unificado = (
    df
    .groupby(["ISSN", "titulo_norm"], as_index=False)
    .agg({
        "titulo": "first",     # conserva un título legible
        "Estrato": unir_unicos,
        "area": unir_unicos
    })
)

# Reordenar columnas
df_unificado = df_unificado[["ISSN", "titulo", "Estrato", "area"]]

# Guardar resultado
df_unificado.to_excel(ruta_salida, index=False)

print("Archivo unificado correctamente:")
print(ruta_salida)
