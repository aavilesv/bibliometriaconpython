import pandas as pd
import re
import unicodedata
import spacy

# Cargar el modelo de spaCy para inglés (para español podrías usar 'es_core_news_sm')
nlp = spacy.load('en_core_web_sm')

def normalizar_texto_spacy(texto, remove_stopwords=False):
    """
    Normaliza un texto en dos etapas:
      1. Limpieza básica:
         - Conversión a minúsculas.
         - Remoción de acentos.
         - Reemplazo de guiones por espacios.
         - Eliminación de paréntesis y su contenido.
         - Conservación de letras, números, espacios y apostrofes.
         - Eliminación de espacios extra.
      2. Procesamiento NLP con spaCy:
         - Tokenización y lematización.
         - Eliminación opcional de stopwords.
    
    Args:
        texto (str): Texto a normalizar.
        remove_stopwords (bool): Si True, elimina las stopwords.
    
    Returns:
        str: Texto normalizado.
    """
    if not texto or not isinstance(texto, str):
        return ""
    
    # Etapa 1: Limpieza básica
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    texto = texto.replace('-', ' ')
    texto = re.sub(r'\([^)]*\)', '', texto)
    texto = re.sub(r"[^a-z0-9'\s]", '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    # Etapa 2: Procesamiento NLP (tokenización y lematización)
    doc = nlp(texto)
    tokens_normalizados = []
    for token in doc:
        if remove_stopwords and token.is_stop:
            continue
        if not token.is_punct and not token.is_space:
            tokens_normalizados.append(token.lemma_)
    return ' '.join(tokens_normalizados)

def normalizar_keywords_columna(celda, remove_stopwords=False):
    """
    Procesa una celda que contiene keywords separadas por ";".
    
    Args:
        celda (str): Cadena de keywords separadas por ";"
        remove_stopwords (bool): Si True, elimina stopwords.
        
    Returns:
        str: Cadena con cada keyword normalizada y reensamblada con ";"
    """
    if not celda or not isinstance(celda, str):
        return ""
    
    # Separar la celda en términos usando ";" como delimitador
    tokens = [token.strip() for token in celda.split(";") if token.strip()]
    # Normalizar cada término individualmente
    tokens_normalizados = [normalizar_texto_spacy(token, remove_stopwords) for token in tokens]
    # Unir de nuevo usando "; " como separador
    return "; ".join(tokens_normalizados)

# 3. Cargar el archivo CSV
ruta_csv = "G:\\Mi unidad\\2025\\Master Kerly Alvarez\\new paper\\data\\wos_scopuslibrería.csv"
df = pd.read_csv(ruta_csv)
# Guardar copia “before”
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
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")

# 4. Aplicar la normalización a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = df['Index Keywords'].apply(lambda x: normalizar_keywords_columna(x, remove_stopwords=False))
df['Author Keywords'] = df['Author Keywords'].apply(lambda x: normalizar_keywords_columna(x, remove_stopwords=False))
# Conteo único después de normalizar
print("\nDespués de normalizar:")
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")
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
#df['Index Keywords'] = eliminar_palabras_clave(df['Index Keywords'])
#df['Author Keywords'] = eliminar_palabras_clave(df['Author Keywords'])
# Opcional: Guardar el DataFrame procesado en un nuevo CSV
df.to_csv("G:\\Mi unidad\\2025\\Master Kerly Alvarez\\new paper\\data\\wos_scopuslibrería_procesado.csv", index=False)
