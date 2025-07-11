import pandas as pd
from collections import Counter
import numpy as np
# Cargar el archivo CSV
ruta =r"G:\\Mi unidad\\2025\\Master Fabre Triana Paula Dominique\\data\\datawos_scopuseliminadas.csv"

df = pd.read_csv(ruta)

# Construir listado global normalizado
all_terms = pd.concat([
    df['Index Keywords'].dropna(),
    df['Author Keywords'].dropna()
]).str.split(';').explode().str.strip()

# Frecuencia sobre versión normalizada
freq = all_terms.str.lower().value_counts()
keep_norm = set(freq[freq > 1].index)

def filter_unique(cell):
    if not isinstance(cell, str):
        return ""
    terms = [t.strip() for t in cell.split(';') if t.strip()]
    filtered = [t for t in terms if t.lower() in keep_norm]
    return '; '.join(filtered)

#df['Index Keywords'] = df['Index Keywords'].apply(filter_unique)
#df['Author Keywords'] = df['Author Keywords'].apply(filter_unique)

def combinar_sin_repetir(row):
    author = row['Author Keywords'] if isinstance(row['Author Keywords'], str) else ""
    index  = row['Index Keywords']  if isinstance(row['Index Keywords'], str) else ""
    
    raw_terms = (author + ";" + index).split(";")
    seen = set()
    unique = []
    for term in raw_terms:
        t = term.strip()
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return "; ".join(unique)

df['Combined Keywords'] = df.apply(combinar_sin_repetir, axis=1)
    
  # 2. Eliminar la columna 'processed_title'
def contar_unicos(col, df_input):
    # Drop NA, split por “;”, strip y filtrar vacíos
    all_keywords = (
        df_input[col]
        .dropna()
        .str.split(';')
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
    )
    return all_keywords.unique().shape[0]

# Conteo único antes de normalizar
print("Antes de normalizar:")
for col in ['Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")
df['Author Keywords'] = df['Combined Keywords']
df.drop(columns="Combined Keywords", inplace=True)

print("\nDespués de normalizar:")
for col in ['Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")
# Guardar el DataFrame resultante en un nuevo archivo CSV

ruta_guardado = r"G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\keywordsunidaswos_scopus.csv"
df.to_csv(ruta_guardado, index=False)

print("Archivo guardado con la columna 'Author Keywords' actualizada y sin repeticiones en:", ruta_guardado)
