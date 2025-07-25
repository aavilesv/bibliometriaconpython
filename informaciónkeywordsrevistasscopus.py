import pandas as pd

# 1. Leer el archivo Excel original
df = pd.read_excel('C:\\Users\\INVESTIGACION 47\\Downloads\\mpdidata.xlsx')  # ajusta el nombre de tu archivo si es distinto

# 2. Asegurar que la columna exista y manejar valores nulos
df['Author Keywords'] = df['Author Keywords'].fillna('').astype(str)

# 3. Dividir la cadena de keywords en listas
df['Keywords List'] = df['Author Keywords'].str.split(';')

# 4. Explotar la lista para obtener una fila por término
df_exploded = df.explode('Keywords List')

# 5. Limpiar espacios en blanco y eliminar filas vacías
df_exploded['Keywords List'] = df_exploded['Keywords List'].str.strip()
df_exploded = df_exploded[df_exploded['Keywords List'] != '']

# 6. Seleccionar y renombrar columnas
result = df_exploded[['Source title', 'Keywords List', 'ISSN']].rename(
    columns={'Keywords List': 'Author Keyword'}
)

# 7. Guardar el resultado en un nuevo Excel
result.to_excel('C:\\Users\\INVESTIGACION 47\\Downloads\\output.xlsx', index=False)

print("Archivo 'output.xlsx' generado con éxito.")
