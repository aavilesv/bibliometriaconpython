import pandas as pd
import re

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv(r"G:\\Mi unidad\\2025\\Master Italo Palacios\\articulo\\datawos_scopus_keywords_norm.csv")

# Diccionario de palabras clave a reemplazar: clave = palabra a buscar (en minúsculas), valor = palabra de reemplazo
palabras_clave_reemplazo = {
    "model of corporate social responsibility": "corporate social responsibility",
   "standard of corporate social responsibility": "corporate social responsibility",
   "forest development project": "forestry development project",
   "nursery": "tree nursery",
   "nursery ground": "tree nursery",
   "rural development programme": "rural development program",
   "academic program": "education program",
   "educational program": "education program",
   "forestry policy": "forest policy",
   "natural forest protection and restoration for": "fnatural forest protection and restoration",
   "forest landscape restoration": "landscape restoration",
   "fforestry education": "forest education",
   "environmental education be": "environmental education",
   "sustainability landscape design":"landscape sustainability",
   "college student":"university student",
   "college building":"university campus",
   "university sector":"university campus",
   "green university campus":"university campus",
   "sustainability landscape design":"landscape sustainability",

   "decarbonising":"decarbonizing",
   "sociol ecological land system":"sociol ecological system",



}

def reemplazar_palabras_clave(column, diccionario_reemplazo):
    """
    Recorre cada celda de la columna, separa los términos (suponiendo que estén separados por ';'),
    y reemplaza aquellos que coincidan (ignorando mayúsculas/minúsculas) por el valor correspondiente del diccionario.
    """
    def process_cell(cell):
        # Si la celda es una cadena, separamos usando el delimitador ';'
        if isinstance(cell, str):
            terminos = [termino.strip() for termino in cell.split(';') if termino.strip()]
        # Si ya es una lista, la usamos directamente
        elif isinstance(cell, list):
            terminos = [str(termino).strip() for termino in cell if str(termino).strip()]
        else:
            terminos = []
        
        terminos_modificados = []
        for termino in terminos:
            # Convertimos el término a minúsculas para la comparación
            termino_lower = termino.lower()
            if termino_lower in diccionario_reemplazo:
                # Reemplazamos por el valor definido en el diccionario
                terminos_modificados.append(diccionario_reemplazo[termino_lower])
            else:
                terminos_modificados.append(termino)
        return '; '.join(terminos_modificados)
    
    return column.apply(process_cell)

# Aplicar la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = reemplazar_palabras_clave(df['Index Keywords'], palabras_clave_reemplazo)
df['Author Keywords'] = reemplazar_palabras_clave(df['Author Keywords'], palabras_clave_reemplazo)
#df['bothKeywords'] =  reemplazar_palabras_clave(df['bothKeywords'], palabras_clave_reemplazo)
# --- A PARTIR DE AQUÍ, EL CÓDIGO NUEVO PARA REEMPLAZOS PARCIALES ---

def reemplazar_parciales(column, patrones):
    """
    Recorre cada celda de la columna, y por cada patrón (regex) en 'patrones',
    realiza la sustitución indicada.
    - 'patrones' debe ser una lista de tuplas (pattern, replacement).
    - Se ignoran mayúsculas/minúsculas (flags=re.IGNORECASE).
    """
    def process_cell(cell):
        if isinstance(cell, str):
            # Aplica todos los patrones de reemplazo parcial
            for patron, nuevo_texto in patrones:
                cell = re.sub(patron, nuevo_texto, cell, flags=re.IGNORECASE)
        return cell

    return column.apply(process_cell)

# Ejemplo de un arreglo de reemplazos parciales
# Cada tupla es (expresión_regular, texto_reemplazo)
# Aquí solo se incluye 'datum' -> 'data', pero puedes añadir más.
patrones_parciales = [
    (r'datum', 'data'),  # Reemplaza 'datum' donde aparezca (ignora mayúsculas)
    # Si necesitas más reemplazos parciales:
    # (r'algunaSubcadena', 'otroTexto'),
    # (r'pattern', 'replacement'),
    # ...
]
# 2) Después, los reemplazos parciales:
#df['bothKeywords'] = reemplazar_parciales(df['bothKeywords'], patrones_parciales)
df['Index Keywords'] = reemplazar_parciales(df['Index Keywords'], patrones_parciales)
df['Author Keywords'] = reemplazar_parciales(df['Author Keywords'], patrones_parciales)
# Guardar el DataFrame modificado en un nuevo archivo CSV
#df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)

df.to_csv(r"G:\\Mi unidad\\2025\\Master Italo Palacios\\articulo\\wos_scopus_reemplazado.csv", index=False)
print("Palabras clave reemplazadas y nuevo archivo guardado.")