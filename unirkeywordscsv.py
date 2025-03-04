import pandas as pd

# Cargar el archivo CSV
ruta =  "G:\\Mi unidad\\2025\\Master Kerly Alvarez\\data\\wos_scopuslibrería_procesado.csv"
df = pd.read_csv(ruta)

# Función para combinar y eliminar duplicados, manejando valores nulos
def combinar_sin_repetir(row):
    # Obtener las palabras clave de ambas columnas, manejando nulos
    author_keywords = row['Author Keywords'] if pd.notna(row['Author Keywords']) else ""
    index_keywords = row['Index Keywords'] if pd.notna(row['Index Keywords']) else ""
    
    # Dividir los valores por ';' y eliminar espacios en blanco
    keywords = set(map(str.strip, author_keywords.split(';') + index_keywords.split(';')))
    
    # Eliminar cualquier cadena vacía del conjunto de keywords
    keywords.discard('')
    
    # Unir los valores únicos en una sola cadena
    return '; '.join(keywords)

# Aplicar la función a cada fila del DataFrame y guardar el resultado en "Author Keywords"
df['Author Keywords'] = df.apply(combinar_sin_repetir, axis=1)

# Guardar el DataFrame resultante en un nuevo archivo CSV
ruta_guardado =  "G:\\Mi unidad\\2025\\Master Kerly Alvarez\\data\\wos_scopuskeywords.csv"
df.to_csv(ruta_guardado, index=False)

print("Archivo guardado con la columna 'Author Keywords' actualizada y sin repeticiones en:", ruta_guardado)
