import pandas as pd
import re

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")
df = pd.read_csv("G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopuslibrería_procesadovos.csv")

# Diccionario de palabras clave a reemplazar: clave = palabra a buscar (en minúsculas), valor = palabra de reemplazo
palabras_clave_reemplazo = {
    "industry 4":"industry 4.0",
    "industry":"industry 4.0",
    "industry 40":"industry 4.0",
    "industry40": "industry 4.0",
    "anylogic industry 40": "industry 4.0",
    "digital economy industry": "digital economy",
    "platform": "digital platform",
    "digital platform industry": "digital platform",
    "supply chain logistic": "supply chain logistic industry 40",
    "open source" : "open source software",
    
    "industry 50" : "industry 5.0",
    
    "industrial chain":"industrial supply chain",
    "industrial engineering": "industrial information integration engineering",
    "industrialisation":"industrialization",
        #"human": "reemplazo_human",
        
        "digitalisation" : "digitalization",
        "digitization" : "digitalization",
        "big datum" : "big data analysis",
        "big datum analytic" : "big data analysis",
        "datum analytic" : "big data analysis",
       "decision make":"decision making",
       "optimisation":"optimization",
       "internet of thing": "internet of things",
       "industrial internet of thing	": "internet of things",
       "industrial iot": "internet of things",
       "industrial internet": "internet of things",
       "internet of thing technology": "internet of things",
       
       "iot": "internet of things",
       "electronic commerce": "e-commerce",
       "commerce adoption":  "e-commerce",
       "e commerce logistic":  "e-commerce",
       "commerce platform":  "e-commerce",
       "e commerce adoption":  "e-commerce",
       "e commerce platform":  "e-commerce",
       "international e commerce":  "e-commerce",
       "food supply": "food supply chain",
       "e commerce adoption" :"e-commerce",
       "e commerce": "e-commerce",
       "electronic commerce": "e-commerce",
       "commerce": "e-commerce",
       "ict": "information and communication technology",
       "rfid": "radio frequency identification",
       "rfid technology": "radio frequency identification technology",
       "radio frequency identification": "radio frequency identification",
       "logistic": "logistics",
       "ai": "artificial intelligence",
       "database factual": "factual database",
       "supply chain 40" : "supply chain",
       "supply chain 4": "supply chain",
       "650 supply chain management": "supply chain management",
       "supply chain digitalization": "digitalization of the supply chain",
       "digitalization of supply chain": "digitalization of the supply chain",
       "5 g" : "5 g network",
       "3 d printing": "3d printing",
       "3d printer": "3d printing",
       "3d":"3d printing",
       "webs service": "web service",
       "industrial technology" :"industry 4.0 technology",
        "industry 40 technology":"industry 4.0 technology",
        "40 technology":"industry 4.0 technology",
        "technology 40":"industry 4.0 technology",
        "smart technology":"smart digital technology",
        "process datum":"data processing",
        "datum processing":"data processing",

       


       

       



    
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
df.to_csv("G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopus_reemplazado.csv", index=False)
print("Palabras clave reemplazadas y nuevo archivo guardado.")