import pandas as pd
import re

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv("G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\datawos_scopuslematizafinal.csv")

# Diccionario de palabras clave a reemplazar: clave = palabra a buscar (en minúsculas), valor = palabra de reemplazo
palabras_clave_reemplazo = {
    "ai": "artificial intelligence",
   "artificial intelligence ai": "artificial intelligence",
   "ai artificial intelligence": "artificial intelligence",
   "artificial intelligent": "artificial intelligence",
   "ai technology": "artificial intelligence technologies",
   "artificial intelligence technology": "artificial intelligence technologies",
   
   "iot": "internet of things",
   "internet of thing": "internet of things",
   "internet of thing iot": "internet of things",
   "internet of thing technology": "internet of things",
   "artificial intelligence of thing": "artificial intelligence of things",
   "aiot": "artificial intelligence of things",
   
   "iiot": "industrial internet of things",
   "industrial internet of thing iiot": "industrial internet of things",
   "industrial internet of thing": "industrial internet of things",

   "ml": "machine learning",
   "machine learning ml": "machine learning",
   "machine learning technique": "machine learning",
   "machine learning model": "machine learning",

   "dl": "deep learning",
   "deep learning dl": "deep learning",

   "cnn": "convolutional neural network",
   "convolutional neural network cnn": "convolutional neural network",
   
   "rpa": "robotic process automation",

   "uav": "unmanned aerial vehicles",
   "unmanned aerial vehicle": "unmanned aerial vehicles",
   "unmanned aerial vehicle uav": "unmanned aerial vehicles",
   "unmanned vehicle": "unmanned aerial vehicles",
   "autonomous unmanned aerial vehicle": "unmanned aerial vehicles",
   "aerial vehicle": "unmanned aerial vehicles",
   "drone": "unmanned aerial vehicles",

   "its": "intelligent transportation systems",
   "intelligent transportation system its": "intelligent transportation systems",
   "intelligent transport system": "intelligent transportation systems",
   "intelligent transport": "intelligent transportation systems",
   "intelligent transportation": "intelligent transportation systems",

   "ids": "intrusion detection system",
   "intrusion detection system ids": "intrusion detection system",
   
   "ict": "information and communication technology",
   "information communication technology": "information and communication technology",

   "bim": "building information modeling",
   "building information modelling": "building information modeling",
   "building information modelling bim": "building information modeling",

   # --- Ortografía, Puntuación y Variaciones Menores ---
   "block chain": "blockchain",
   "cyber security": "cybersecurity",
   "cyber-physical system cp": "cyber-physical systems",
   "cybe physical system": "cyber-physical systems",
   "cyber physical system": "cyber-physical systems",
   "optimisation": "optimization",
   "digitisation": "digitization",
   "digitalisation": "digitization",
   "e government": "e-government",
   "e governance": "e-government",
   "digital government": "e-government",
   "real time": "real-time",
   "real- time": "real-time",
   "real time system": "real-time systems",
   "covid-19": "coronavirus",
   "sar cov-2": "coronavirus",
   "5 g": "5g",
   "5 g mobile communication system": "5g",
   "industry 40": "industry 4.0",
   "fourth industrial revolution": "industry 4.0",
   "industry 50": "industry 5.0",
   "wi fi": "wifi",
   "wireless fidelity": "wifi",
   
   # --- Sinónimos y Normalización Jerárquica ---
   "public sector": "public administration",
   "local government": "public administration",
   "government": "public administration",
   "automate decision making": "automated decision making",
   "automate decision making system": "automated decision making",
   "decision making process": "decision making",
   "smart city": "smart cities",
   "sustainable city": "smart cities",
   "digital city": "smart cities",
   "urban development": "urban planning",
   "city planning": "urban planning",
   "transport": "transportation",
   "transport system": "transportation systems",
   "transportation system": "transportation systems",
   "car": "automobiles",
   "automobile": "automobiles",
   "automate vehicle": "autonomous vehicles",
   "autonomous vehicle": "autonomous vehicles",
   "sustainability": "sustainable development",
   "deep neural network": "neural networks",
   "artificial neural network": "neural networks",
   "ann artificial neural network": "neural networks",
   "recurrent neural network": "neural networks",
   "convolutional neural network": "neural networks",
   "neural network": "neural networks",
   "data analytic": "data analytics",
   "big data analytic": "big data analytics",
   "healthcare": "health care",
   "delivery of health care": "health care",
   "smart grid": "smart grids",
   "smart power grid": "smart grids",
   "intelligent building": "smart buildings",
   "smart building": "smart buildings",

   # --- Estandarización a Plural (siguiendo tu ejemplo) ---
   "algorithm": "algorithms",
   "learning algorithm": "algorithms",
   "classification algorithm": "algorithms",
   "system": "systems",
   "intelligent system": "intelligent systems",
   "learning system": "learning systems",
   "embed system": "embedded systems",
   "technology": "technologies",
   "digital technology": "technologies",
   "emerging technology": "technologies",
   "automation technology": "automation technologies",
   "review": "reviews",
   "literature review": "reviews",
   "systematic review": "reviews",
   "systematic literature review": "reviews",
   "model": "models",
   "learning model": "models",
   "predictive model": "models",
   "challenge": "challenges",
   "research challenge": "challenges",
   "public service": "public services",
   "human": "humans",
   "law and legislation": "laws",
   "regulation": "laws",
   "legislation": "laws",
   "building": "buildings",
   "city": "cities",
   "urban area": "urban areas",


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

df.to_csv("G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\wos_scopus_reemplazado.csv", index=False)
print("Palabras clave reemplazadas y nuevo archivo guardado.")