import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("G:\\Mi unidad\\2024\\Master JAVIER ANDRES CHILIQUINGA AMAYA\\IDH Bibliometría\\data\\wos_scopus.csv")

   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [
  "human", "Humans", "Female", "Male", "controlled study", "Adult", "major clinical study", "0", "sensitivity and specificity",

    "image reconstruction", "positron emission tomography"
]

# Función para eliminar palabras clave específicas y retornar una cadena
def eliminar_palabras_clave(column):
    # Convertir la lista de palabras clave a eliminar a minúsculas
    palabras_clave_a_eliminar_lower = [palabra.lower() for palabra in palabras_clave_a_eliminar]
    
    def process_cell(cell):
        # Si la celda es una cadena, la dividimos en una lista usando el separador ';'
        if isinstance(cell, str):
            terminos = [termino.strip() for termino in cell.split(';') if termino.strip()]
        # Si ya es una lista, la usamos directamente
        elif isinstance(cell, list):
            terminos = [str(termino).strip() for termino in cell if str(termino).strip()]
        else:
            terminos = []
        # Filtrar los términos que, al pasar a minúsculas, estén en la lista a eliminar
        terminos_filtrados = [termino for termino in terminos if termino.lower() not in palabras_clave_a_eliminar_lower]
        # Unir la lista filtrada en una cadena usando '; ' como separador
        return '; '.join(terminos_filtrados)

    return column.apply(process_cell)

# Aplicar la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = eliminar_palabras_clave(df['Index Keywords'])
df['Author Keywords'] = eliminar_palabras_clave(df['Author Keywords'])
# Guardar el DataFrame filtrado en un nuevo archivo CSV

# Guardar el DataFrame filtrado en un nuevo archivo CSV
df.to_csv("G:\\Mi unidad\\2024\\Master JAVIER ANDRES CHILIQUINGA AMAYA\\IDH Bibliometría\\data\\wos_scopus.csv", index=False)

print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
