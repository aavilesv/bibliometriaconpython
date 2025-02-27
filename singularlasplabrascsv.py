import pandas as pd
import nltk
import unicodedata
import spacy  # Para análisis sintáctico (reordenamiento dinámico)

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re  # Para normalizar los espacios

# 1. Descargar los recursos necesarios de NLTK
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# 2. Inicializar el lematizador de WordNet
lemmatizer = WordNetLemmatizer()
# 3. Cargar el modelo de spaCy para inglés (asegúrate de haberlo descargado con "python -m spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
# Función para convertir etiquetas POS de NLTK (Treebank) a etiquetas que entiende WordNet
def get_wordnet_pos(treebank_tag):
    """
    Convierte una etiqueta del Treebank a una etiqueta compatible con WordNet.
    Si la etiqueta no es de las reconocidas, por defecto se considera como sustantivo.
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ      # Adjetivo
    elif treebank_tag.startswith('V'):
        return wn.VERB     # Verbo
    elif treebank_tag.startswith('N'):
        return wn.NOUN     # Sustantivo
    elif treebank_tag.startswith('R'):
        return wn.ADV      # Adverbio
    else:
        return wn.NOUN     # Por defecto, sustantivo

# Función para lematizar una palabra a su forma base (singular para sustantivos)
def to_singular(word):
    """
    Lematiza una palabra utilizando su etiqueta POS para obtener la forma base.
    Esto permite que, por ejemplo, "humans" se convierta en "human".
    """
    if isinstance(word, str):
        # Obtiene la etiqueta POS de la palabra
        pos = pos_tag([word])[0][1]
        # Convierte la etiqueta a la que espera WordNet
        wordnet_pos = get_wordnet_pos(pos)
        # Lematiza la palabra con la etiqueta correspondiente
        return lemmatizer.lemmatize(word, pos=wordnet_pos)
    return word

def limpiar_texto(texto):
    """
    Limpia el texto eliminando:
      - Guiones (-), reemplazándolos por un espacio.
      - Paréntesis y todo lo que esté dentro de ellos.
      - Cualquier carácter especial, dejando solo letras y espacios.
      - Además, convierte a minúsculas y remueve acentos.
    """
    # Reemplaza guiones por espacios
    texto = texto.replace('-', ' ')
    # Elimina paréntesis y su contenido
    texto = re.sub(r'\([^)]*\)', '', texto)
    # Elimina caracteres especiales, dejando solo letras y espacios
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)
    # Convierte a minúsculas
    texto = texto.lower()
    # Normaliza texto removiendo acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Elimina espacios extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def normalizar_frase(frase):
    """
    Utiliza spaCy para analizar la frase y, si se identifica un chunk nominal
    que abarque casi toda la frase, reordena sus componentes para colocar
    el núcleo (head) al final.
    
    Por ejemplo, si spaCy identifica en "nurse public health" un chunk nominal
    cuyo núcleo es "nurse" y modificadores "public" y "health", reconstruye la
    frase como "public health nurse".
    """
    doc = nlp(frase)
    chunks = list(doc.noun_chunks)
    if not chunks:
        return frase  # Si no hay chunks nominales, se retorna la frase sin modificar
    # Busca un chunk que abarque toda la frase (o casi)
    for chunk in chunks:
        if chunk.start == 0 and chunk.end == len(doc) and len(chunk) > 1:
            head = chunk.root
            modifiers = [token for token in chunk if token != head]
            nueva_frase = " ".join([token.text for token in modifiers] + [head.text])
            return nueva_frase
    return frase
# Función para procesar cada término (ya sea una sola palabra o una frase compuesta)
def procesar_termino(termino):
    """
    Procesa un término (ya sea una sola palabra o una frase compuesta) realizando:
      1. Limpieza del texto (eliminación de guiones, paréntesis y caracteres especiales).
      2. División en palabras (eliminando espacios extra).
      3. Lematización de cada palabra.
      4. Unión de las palabras con un único espacio.
      5. Reordenamiento dinámico (usando spaCy) para normalizar la estructura del chunk nominal.
    """
    # Limpia el texto
    termino_limpio = limpiar_texto(termino)
    # Separa la frase en palabras (split elimina espacios extras)
    palabras = termino_limpio.split()
    # Lematiza cada palabra
    palabras_lemmatizadas = [to_singular(palabra.strip()) for palabra in palabras]
    # Une las palabras lematizadas
    frase_lemmatizada = " ".join(palabras_lemmatizadas)
    # Asegura que solo haya un espacio entre palabras
    frase_lemmatizada = re.sub(r'\s+', ' ', frase_lemmatizada).strip()
    # Aplica reordenamiento dinámico si es aplicable
    frase_normalizada = normalizar_frase(frase_lemmatizada)
    return frase_normalizada


# Función para aplicar la lematización a toda una columna de un DataFrame
def convertir_palabras_a_singular(column):
    """
    Toma una columna del DataFrame, limpia valores nulos y convierte el texto a minúsculas.
    Luego, separa cada término (asumiendo que están separados por ';'), procesa cada uno
    lematizando todas las palabras y vuelve a unirlos.
    """
    # Rellena valores NaN y convierte todo a minúsculas
    column = column.fillna('').str.lower()
    
    # Separa cada celda por ';', procesa cada término y vuelve a unirlos
    palabras_procesadas = column.str.split(';').apply(
        lambda terminos: '; '.join([procesar_termino(termino.strip()) for termino in terminos])
    )
    return palabras_procesadas

# 3. Cargar el archivo CSV y aplicar la conversión
df = pd.read_csv("G:\\Mi unidad\\2025\\Dra. Maria Caro\\data\\wos_scopuslibrería_keywords.csv")

# Aplica la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = convertir_palabras_a_singular(df['Index Keywords'])
df['Author Keywords'] = convertir_palabras_a_singular(df['Author Keywords'])


# Opcional: Guardar el DataFrame procesado en un nuevo CSV
df.to_csv("G:\\Mi unidad\\2025\\Dra. Maria Caro\\data\\wos_scopuslibrería_procesado.csv", index=False)
'''

import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Descargar el paquete WordNet de NLTK si aún no está instalado
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Inicializa el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar el archivo CSV
df = pd.read_csv("G:\\Mi unidad\\2025\\Dra. Maria Caro\\data\\wos_scopuslibrería.csv")

# Función para convertir palabras a singular usando el lematizador
def to_singular(word):
    if isinstance(word, str):
        return lemmatizer.lemmatize(word)
    return word  # Devuelve la palabra original si no es un `str`

# Función para procesar términos (manteniendo frases intactas si tienen más de una palabra)
def convertir_palabras_a_singular(column):
    # Rellenar valores NaN con cadena vacía y convierte a minúsculas
    column = column.fillna('').str.lower()
    
    def procesar_terminos(termino):
        palabras = termino.split()  # Divide términos en palabras
        # Si es una sola palabra, lematiza. Si es una frase, la deja igual.
        if len(palabras) == 1:
            return to_singular(palabras[0])
        return termino.strip()  # Mantiene frases intactas

    # Divide en listas por separador ";", procesa cada término, y une de nuevo
    palabras_procesadas = column.str.split(';').apply(
        lambda terminos: '; '.join([procesar_terminos(termino.strip()) for termino in terminos])
    )
    return palabras_procesadas

# Aplicar la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = convertir_palabras_a_singular(df['Index Keywords'])
df['Author Keywords'] = convertir_palabras_a_singular(df['Author Keywords'])


# Combinar palabras de ambas columnas en un solo contador
palabras = df['Index Keywords'].explode().tolist()+ ";"+ df['Author Keywords'].explode().tolist()
frecuencias = Counter(palabras)

# Filtrar palabras únicas (frecuencia = 1)
palabras_unicas = {palabra for palabra, freq in frecuencias.items() if freq == 1}

# Eliminar palabras únicas del DataFrame
def eliminar_palabras_unicas(column):
    return column.apply(
        lambda terminos: [termino for termino in terminos if termino not in palabras_unicas]
    )



# Función para agrupar sinónimos usando WordNet
def agrupar_sinonimos(column):
    def obtener_sinonimo_base(termino):
        # Buscar sinónimos en WordNet
        sinonimos = wn.synsets(termino)
        if sinonimos:
            # Retornar la palabra base del primer sinónimo
            return sinonimos[0].lemmas()[0].name()
        return termino  # Si no hay sinónimo, devuelve el término original

    # Reemplazar cada término por su sinónimo base
    return column.apply(
        lambda terminos: [obtener_sinonimo_base(termino) for termino in terminos]
    )

# Agrupar sinónimos en ambas columnas
df['Index Keywords'] = agrupar_sinonimos(df['Index Keywords'])
df['Author Keywords'] = agrupar_sinonimos(df['Author Keywords'])
df['Index Keywords'] = eliminar_palabras_unicas(df['Index Keywords'])
df['Author Keywords'] = eliminar_palabras_unicas(df['Author Keywords'])
# Guardar el DataFrame procesado en un nuevo archivo CSV
df['Index Keywords'] = df['Index Keywords'].apply('; '.join)
df['Author Keywords'] = df['Author Keywords'].apply('; '.join)

# Guardar el DataFrame procesado en un nuevo archivo CSV
df.to_csv("G:\\Mi unidad\\2024\\Master Solis Granda Luis Eduardo\\data\\wos_scopuskeywordsfinal.csv", index=False)

print("Palabras clave convertidas a singular (si corresponde) y nuevo archivo guardado.")
'''