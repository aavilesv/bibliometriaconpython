import pandas as pd
from collections import Counter
import numpy as np
# Cargar el archivo CSV
ruta = "G:\\Mi unidad\\2025\\master kevin castillo\\artículo nuevo\\data\\datawos_scopuslematizar.csv"
       

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

df['Author Keywords'] = df['Combined Keywords']
df.drop(columns="Combined Keywords", inplace=True)

# Guardar el DataFrame resultante en un nuevo archivo CSV

ruta_guardado = "G:\\Mi unidad\\2025\\master kevin castillo\\artículo nuevo\\data\\datawos_scopuskeywords.csv"
df.to_csv(ruta_guardado, index=False)

print("Archivo guardado con la columna 'Author Keywords' actualizada y sin repeticiones en:", ruta_guardado)
