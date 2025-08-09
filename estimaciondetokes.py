import pandas as pd
import math
import re

# === RUTA DE TU ARCHIVO ===
csv_path = r"G:/Mi unidad/Artículos cientificos/articulo 1/datawos_scopus_limpio.csv"
out_path = csv_path.replace(".csv", "_with_affil_merged.csv")

COL1 = "Affiliations"
COL2 = "Authors with affiliations"
MERGED_COL = "Affil_merged"
LEN_COL = "Affil_merged_charlen"

# Carga
df = pd.read_csv(csv_path)

# Verificación de columnas
for c in (COL1, COL2):
    if c not in df.columns:
        raise ValueError(f"❌ Falta la columna: {c}")

# Función para unir con separador solo si ambos existen
def smart_join(a, b, sep=" ; "):
    a = "" if pd.isna(a) else str(a).strip()
    b = "" if pd.isna(b) else str(b).strip()
    if a and b:
        merged = f"{a}{sep}{b}"
    else:
        merged = a or b  # el que no esté vacío
    # Normaliza espacios y separadores repetidos
    merged = re.sub(r"\s+", " ", merged).strip()
    merged = re.sub(r"\s*;\s*", " ; ", merged)  # separador limpio
    return merged

# Crear columna unida
df[MERGED_COL] = [smart_join(a, b) for a, b in zip(df[COL1], df[COL2])]

# Longitud por fila (caracteres, incluyendo espacios)
df[LEN_COL] = df[MERGED_COL].fillna("").map(len)

# Suma total de caracteres
total_chars = int(df[LEN_COL].sum())

# Estimación de tokens (aprox. para español)
EST_CHARS_PER_TOKEN = 3.5
estimated_tokens = math.ceil(total_chars / EST_CHARS_PER_TOKEN)

print(f"✅ Total de caracteres en '{MERGED_COL}': {total_chars:,}")
print(f"≈ Estimación de tokens (1 tok ≈ {EST_CHARS_PER_TOKEN} chars): {estimated_tokens:,}")

# Guarda
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"📄 Archivo guardado: {out_path}")
