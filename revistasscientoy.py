import pandas as pd


import re
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS  # Stopwords en inglés
from rapidfuzz import fuzz, process

# Cargar el archivo CSV

df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")
#df = pd.read_csv("G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopus_unirkeywords.csv")
# Lista global para almacenar títulos únicos de revistas
unique_titles = []
def process_source_title(title, threshold=90):
    """
    Procesa y normaliza el título de la revista de forma dinámica.
    """
    global unique_titles
    if not isinstance(title, str):
        return ""
    # Limpieza básica del título
    title = title.lower()
    title = re.sub(r'\([^)]*\)', '', title)  # Eliminar texto entre paréntesis
    title = title.replace('-', ' ')            # Reemplazar guiones por espacios
    title = re.sub(r'\s+', ' ', title).strip()  # Eliminar espacios redundantes
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Capitalizar cada palabra (formato título)
    if not title.isupper():
        title = " ".join([word.capitalize() for word in title.split()])
    # Fuzzy Matching para unificar títulos similares
    if unique_titles:
        best_match, score, _ = process.extractOne(title, unique_titles, scorer=fuzz.token_sort_ratio)
        if score > threshold:
            return best_match
    unique_titles.append(title)
    return title
    
df['Source title'] = df['Source title'].apply(process_source_title)

# Guardar el DataFrame filtrado en un nuevo archivo CSV
df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)
#df.to_csv("G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopuslibrería_procesadovos.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
