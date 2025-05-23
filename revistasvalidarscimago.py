import pandas as pd
import numpy as np

# Definir la ruta base
ruta = r'G:\\Mi unidad\\2025\\codigos bibliometria NPL\\'

# Definir la ruta completa de los archivos
archivo_data1 = ruta + r'\\revistas.xlsx'
archivo_data2 = ruta + r'\\ext_list_March_2025.xlsx'

# Cargar ambos archivos Excel
df1 = pd.read_excel(archivo_data1)
df2 = pd.read_excel(archivo_data2)

# Convertir las columnas 'ISSN' y 'EISSN' de df2 a string, reemplazando valores nulos por cadena vacía
df2['ISSN'] = df2['ISSN'].fillna('').astype(str)
df2['EISSN'] = df2['EISSN'].fillna('').astype(str)

def obtener_estado(issn_val):
    # Verificar si el valor es nulo o vacío
    if pd.isna(issn_val) or issn_val == '':
        return None

    # Convertir el valor a string y separar por comas, eliminando espacios extra
    codes = [codigo.strip() for codigo in str(issn_val).split(',') if codigo.strip() != '']
    
    # Recorrer cada código y buscar en df2
    for code in codes:
        # Filtrar las filas de df2 donde ISSN o EISSN coincida con el código
        coincidencia = df2[(df2['ISSN'] == code) | (df2['EISSN'] == code)]
        if not coincidencia.empty:
            # Retornar el primer valor encontrado en 'Active or Inactive'
            return coincidencia.iloc[0]['Active or Inactive']
    # Si no se encuentra ninguna coincidencia, se retorna None
    return "Inactive"

# Aplicar la función a la columna 'Issn' del DataFrame df1 y crear la nueva columna
df1['Active or Inactive'] = df1['Issn'].apply(obtener_estado)

# Guardar el DataFrame actualizado en un nuevo archivo Excel en la misma ruta
archivo_salida = ruta + r'\\data1_actualizado.xlsx'
df1 = df1.drop_duplicates()
df1.to_excel(archivo_salida, index=False)

print(f"Proceso completado. El archivo '{archivo_salida}' se ha guardado con la nueva columna.")
