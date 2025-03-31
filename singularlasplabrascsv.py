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

# Función para lematizar una palabra a su forma base (singular) si es un sustantivo
def to_singular(word):
    """
    Lematiza una palabra utilizando su etiqueta POS para obtener la forma base.
    Si la palabra es un sustantivo en plural, se convierte a singular;
    en otros casos, se devuelve sin cambios.
    """
    if isinstance(word, str):
        # Obtiene la etiqueta POS de la palabra
        pos = pos_tag([word])[0][1]
        if pos.startswith('N'):  # Solo lematiza si es un sustantivo
            return lemmatizer.lemmatize(word, pos='n')
        else:
            return word
    return word

def limpiar_texto(texto):
    """
    Limpia el texto eliminando:
      - Guiones (-), reemplazándolos por un espacio.
      - Paréntesis y todo lo que esté dentro de ellos.
      - Cualquier carácter especial, dejando solo letras y espacios.
      - Además, convierte a minúsculas y remueve acentos.
    """
    # 1 Convierte a minúsculas
    texto = texto.lower()
    
    # 2 Normaliza texto removiendo acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # 3 Reemplaza guiones por espacios
    texto = texto.replace('-', ' ')
    # 4 Elimina paréntesis y su contenido
    texto = re.sub(r'\([^)]*\)', '', texto)
    # 5 Elimina caracteres especiales, dejando solo letras y espacios
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    #Si en algún momento necesitas conservar otros caracteres
    #texto = re.sub(r"[^a-z0-9'\s]", '', texto)

    # 6 Elimina espacios extras
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
df = pd.read_csv("G:\\Mi unidad\\2025\\Master Yindra flores\\data\\wos_scopuslibreríakewyrodrs.csv")
# Aplica la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = convertir_palabras_a_singular(df['Index Keywords'])
df['Author Keywords'] = convertir_palabras_a_singular(df['Author Keywords'])


# Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [
  "human", "Humans", "Female", "Male", "controlled study", "Adult", "major clinical study", "0", "sensitivity and specificity",
    "retrospective study", "receiver operating characteristic", "diagnostic imaging", "predictive value", "diagnostic accuracy",
    "diagnostic test accuracy study", "nuclear magnetic resonance imaging", "image segmentation", "Diseases", "pathology",
    "follow up", "Young Adult", "Social Media", "image analysis", "Adolescent", "Breast cancer", "Magnetic Resonance Imaging",
    "electronic health record", "risk factor", "very elderly", "Electronic Health Records", "clinical article", "clinical feature",
    "prospective study", "radiomics", "nonhuman", "x-ray computed tomography", "Tomography, X-Ray Computed",
    "clinical effectiveness", "laboratory automation", "prognosis", "clinical trial", "diagnostic value", "digital health",
    "physiology", "biological marker", "neuroimaging", "Breast Neoplasms", "Breast", "Alzheimer disease",
    "Architectural design", "diabetes mellitus", "drug industry", "false positive result", "metabolism", "current", "Animals",
    "Hospitals", "clinical decision making", "cross-sectional study", "health care delivery", "longitudinal study", "Child",
    "Automation", "Laboratory", "Comorbidity", "Electronic medical record", "Health", "Radio waves",
    "Least squares approximations", "Leukocyte Count", "Life cycle assessment", "MANUFACTURING FIRMS", "MATURITY",
    "Manufacturing process", "Network architecture", "PARTICLE SWARM OPTIMIZATION", "Particle swarm optimization (PSO)",
    "RESOURCE ORCHESTRATION", "Resource orchestration theory", "Robot programming", "Service Quality", "Smart cities",
    "Smart farming", "Structural equation modelling", "Survey", "Time Factors", "Traffic congestion", "cancer classification",
    "cancer screening", "data", "demographics", "development", "differential diagnosis", "drug", "echography", "ecosystem",
    "educational status", "employment", "environmental economics", "ergonomics", "financing constraints", "food quality",
    "food waste", "fuzzy DEMATEL", "human experiment", "marketing", "modelling", "motivation", "normal human",
    "pharmaceutics", "preoperative evaluation", "prevalence", "time factor", "university hospital",
    "5G mobile communication systems", "AMBIDEXTERITY", "ANALYTIC HIERARCHY PROCESS", "ASSIGNMENT", "Access control",
    "Agile manufacturing systems", "BEHAVIOR", "Biomarkers", "Blood", "COMPETENCE", "CONSTRAINTS", "CRANES", "CUSTOMER",
    "Computation theory", "Configuration", "Coronary artery disease", "Crisis", "DUAL-CHANNEL", "Delphi study",
    "ENERGY MANAGEMENT", "ENVIRONMENTAL UNCERTAINTY", "Electrocardiography", "Errors", "Europe",
    "Flexible manufacturing systems", "Fuzzy AHP", "GENERATION", "HUMAN-RESOURCE MANAGEMENT", "ISM", "Italy", "LEVEL",
    "Large dataset", "Longitudinal Studies", "MARKET ORIENTATION", "Machine design", "Medical imaging", "Model validation",
    "Models, Statistical", "Motion planning", "Multivariate Analysis", "Navigation", "ORIENTATION", "PERSPECTIVES",
    "PRESSURES", "Pharmaceutical industry", "Remote Sensing", "Renewable energy", "Research and development",
    "Surveys and Questionnaires", "TOOLS", "UTAUT", "VARIABLES", "WORK", "agribusiness", "cognitive defect",
    "complexity", "complication", "cost-benefit analysis", "developing world", "echomammography", "energy conservation",
    "entropy", "factual database", "glucose", "glucose blood level", "health service", "hospital pharmacy", "learning",
    "lung cancer", "mammography", "manufacturing industry", "manufacturing systems", "medical record", "mental health",
    "mobile phone", "neoplasm", "qualitative research", "radiologist", "segmentation", "smoking", "Application programs",
    "Biopsy", "Chains", "ELECTRIC VEHICLES", "Fruits", "Game theory", "Lymph Nodes", "METHOD BIAS", "Markov chain",
    "Modeling", "Platform", "STATE", "catering service", "clinical decision support system", "grasping",
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
df.to_csv("G:\\Mi unidad\\2025\\Master Yindra flores\\data\\wos_scopuslibrería_procesado.csv", index=False)