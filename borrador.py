data_csv = r"G:\\Mi unidad\2025\\Master Karla Villavicencio\\taylor correciones\\"


import pandas as pd

# Ruta al archivo CSV
data_csv = r"G:\\Mi unidad\2025\\Master Karla Villavicencio\\taylor correciones\\Implicaciones gerenciales y toma de decisiones.csv"

# Cargar datos desde CSV
df = pd.read_csv(data_csv, dtype=str).fillna('')

# Columnas de interés de Scopus
columnas_busqueda = ['Title', 'Abstract', 'Author Keywords', 'Index Keywords']

# Lista de términos a buscar (ya en minúsculas)	
terminos = [
      
   
     "managerial implications", "business value", "strategic decision making", 
    "return on investment", "competitive advantage", "marketing strategy", 
    "AI governance", "adoption barriers",

    "cost–benefit analysis", "organizational readiness", "change management", 
    "C-level adoption", "KPI"

    ]
# Normalizar todo a minúsculas en las columnas de búsqueda
df[columnas_busqueda] = df[columnas_busqueda].apply(lambda col: col.str.lower())

# Función que verifica si algún término aparece en el texto
def contiene_termino(texto):
    return any(term in texto for term in terminos)

# Crear máscara para filas que contienen al menos un término en cualquiera de las columnas
mask = df[columnas_busqueda].applymap(contiene_termino).any(axis=1)

# Filtrar el DataFrame según la máscara
df_filtrado = df.loc[mask].copy()

# Mostrar cuántas filas cumplen el criterio
print(f"Filas encontradas: {len(df_filtrado)}")

# Guardar en Excel

data_salida = r"G:\\Mi unidad\2025\\Master Karla Villavicencio\\taylor correciones\\Implicaciones gerenciales y toma de decisionesresultados_scopus_filtrados.xlsx"
with pd.ExcelWriter(data_salida, engine='openpyxl') as writer:
    df_filtrado.to_excel(writer, index=False, sheet_name='Filtrados')
print(f"Archivo guardado en: {data_salida}")
