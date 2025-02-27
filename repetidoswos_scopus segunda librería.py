#pip install rapidfuzz
#pip install fuzzywuzzy python-Levenshtein

import pandas as pd
import spacy
import re
import unicodedata
from spacy.lang.en.stop_words import STOP_WORDS  # Stopwords en inglés
from rapidfuzz import fuzz, process
try:
    # Cargar modelo de spaCy en español
    nlp = spacy.load('en_core_web_lg')
    
    # Función para preprocesar los títulos
    #ejemplo de la función 
    #title = "Análisis de la percepción (2023) sobre el clima, en ciudades latinoamericanas!"
    #print(preprocess_title(title))
    #salida
    #analisis percepcion clima ciudades latinoamericanas

    def preprocess_title(title):
        # Asegurar que el título no sea NaN o None
        if not isinstance(title, str):
            return ""
        
        # Eliminar caracteres especiales excepto letras y espacios
        title = re.sub(r"[^a-zA-Z\s]", "", title)
        
        # Convertir a minúsculas
        title = title.lower()
        
        # Normalizar texto (remover acentos)
        title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # Eliminar palabras comunes (stopwords) en inglés
        words = title.split()
        filtered_words = [word for word in words if word not in STOP_WORDS]
        title = " ".join(filtered_words)
        
        # Eliminar múltiples espacios
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title


    # Cargar los datos
    
    scopus_file_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\clusting o hj biplot\\data_scopus.csv"
    
    wos_file_path ="G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\clusting o hj biplot\\data_wos.xlsx"
    try:
        # Leer los datos del archivo CSV de Scopus
        scopus_df = pd.read_csv(scopus_file_path)
        
        # Leer los datos del archivo Excel de Web of Science
        wos_df = pd.read_excel(wos_file_path)
        wos_df['Authors'] = wos_df['Authors'].str.replace(',', '')
    except Exception as e:
        print(f"Error al cargar los archivos: {e}")
        raise
 
    # Preprocesar los títulos en ambos dataframes
    scopus_df['processed_title'] = scopus_df['Title'].apply(preprocess_title)
    wos_df['processed_title'] = wos_df['Article Title'].apply(preprocess_title)
    # Valores iniciales
    print("**Valores iniciales:**")
    print(f"Total de artículos en Scopus: {len(scopus_df)}")
    print(f"Total de artículos en Web of Science: {len(wos_df)}\n")
    # Contar los duplicados antes de eliminarlos
    scopus_duplicates_count = scopus_df.duplicated(subset=['processed_title']).sum()
    wos_duplicates_count = wos_df.duplicated(subset=['processed_title']).sum()

    print(f"Duplicados en Scopus antes de eliminar: {scopus_duplicates_count}")
    print(f"Duplicados en Web of Science antes de eliminar: {wos_duplicates_count}\n")

    # Eliminar duplicados dentro de cada dataframe basado en los títulos procesados
    scopus_df = scopus_df.drop_duplicates(subset=['processed_title'])
    wos_df = wos_df.drop_duplicates(subset=['processed_title'])
    scopus_df['DOI'] = scopus_df['DOI'].fillna('').str.lower().str.strip()
    wos_df['DOI'] = wos_df['DOI'].fillna('').str.lower().str.strip()
    # (1) Primero, coincidencias por DOI
    doi_matches = []
    scopus_dois = set(scopus_df['DOI'].values)  # para búsquedas rápidas

    for idx, wos_row in wos_df.iterrows():
        wos_doi = wos_row['DOI']
        if wos_doi and wos_doi in scopus_dois:
            # DOI coincide => duplicado seguro
            doi_matches.append(wos_row['processed_title'])

    # (2) Fuzzy matching para títulos
    # definimos un umbral (por ejemplo, 90)
    threshold_fuzzy = 90
    similar_titles = []

    # Convertimos los títulos de Scopus en lista para fuzzy
    scopus_titles_list = scopus_df['processed_title'].tolist()

    for idx, wos_row in wos_df.iterrows():
        wos_title = wos_row['processed_title']
        wos_doi = wos_row['DOI']

        # Si ya se detectó duplicado por DOI, saltamos
        if wos_title in doi_matches:
            continue

        # Si el DOI coincide, ya está en doi_matches
        if wos_doi and wos_doi in scopus_dois:
            doi_matches.append(wos_title)
            continue
        
        # Fuzzy matching
        # Retorna (best_match, score, match_index)
        best_match, score, _ = process.extractOne(
            wos_title,
            scopus_titles_list,
            scorer=fuzz.token_sort_ratio
        )
        
        if score > threshold_fuzzy:
            similar_titles.append(wos_title)

    # Combinar las listas (puedes unificarlas si quieres)
    all_duplicates = set(doi_matches + similar_titles)

    print(f"Duplicados detectados por DOI: {len(doi_matches)}")
    print(f"Duplicados detectados por fuzzy: {len(similar_titles)}")

    # Total de duplicados
    print(f"n total hay {len(scopus_df) + len(wos_df)} artículos, En total hay {len(all_duplicates)} artículos repetidos.")
   


     # --- 5) Guardar los títulos repetidos en un archivo CSV ---
    output_file_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\clusting o hj biplot\\wos_scopus_repeatedstitles.csv"
    repeated_titles_df = pd.DataFrame(list(all_duplicates), columns=['Título Repetido'])
  
    
    try:
  
        repeated_titles_df.to_csv(output_file_path, index=False)
        print("Los títulos repetidos han sido guardados en 'wos_scopus_repeatedstitles.csv'.")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")

 
    # --- 6) Eliminar los títulos repetidos en wos_df y scopus_df ---
    wos_non_repeated = wos_df[~wos_df['processed_title'].isin(all_duplicates)]
    

    
    df_wos_renombrado = wos_non_repeated.rename(columns={
                'Authors': 'Authors',
                'Document Type': 'Document Type',
                'Language': 'Language of Original Document',
                'Author Keywords': 'Author Keywords',
                'Keywords Plus': 'Index Keywords',
                'Abstract': 'Abstract',
                'DOI': 'DOI',
                'Author Full Names':'Author full names',
                'Cited Reference Count': 'Cited by',
                'Publication Year': 'Year',
                'Source Title': 'Source title',
                'Article Title': 'Title',
                'Addresses':'Authors with affiliations',
                'Open Access Designations':'Open Access',
                'ISSN': 'ISSN',
                'Publisher': 'Publisher',
                'DOI Link': 'Link'
                
                
            })

     # Seleccionar solo las columnas necesarias en Web of Science
    necessary_columns = [
        'Authors', 'Document Type', 'Language of Original Document', 'Author Keywords', 
        'Index Keywords', 'Abstract', 'DOI', 'Cited by', 'Year', 'Source title', 
        'Title', 'Affiliations', 'ISSN', 'Publisher', 'Link','Open Access', 'Author full names','Scopus_SubjectArea', 
        'Authors with affiliations', 'processed_title'
    ]
    
    # Validar que las columnas existan en df_wos_renombrado
    final_cols = [col for col in necessary_columns if col in df_wos_renombrado.columns]
    df_wos_renombrado = df_wos_renombrado[final_cols]
    
    # --- 8) Combinar datos ---
    # Unimos scopus_no_repeated y wos_no_repeated en un DataFrame final
    combined_df = pd.concat([scopus_df, df_wos_renombrado], ignore_index=True)
    def process_authors(authors):
        if not isinstance(authors, str):
            return ""
    
        # Reemplazar guiones por espacios
        authors = authors.replace('-', ' ')
    
        # Eliminar puntos pero preservar las iniciales completas
        authors = re.sub(r'\b([A-Z])\.', r'\1', authors)
    
        # Normalizar texto (remover acentos)
        authors = unicodedata.normalize('NFKD', authors).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
        # Capitalizar cada palabra
        authors = " ".join([word.capitalize() for word in authors.split()])
    
        return authors
    # Guardar el DataFrame combinado en un archivo CSV
    combined_output_file_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\clusting o hj biplot\\wos_scopuslibrería.csv"
    try:
        combined_df.to_csv(combined_output_file_path, index=False)
        # Resultados finales
        print("**Resultados finales:**")
        print(f"Total de artículos únicos combinados: {len(combined_df)}")
        
        print("Los datos combinados han sido guardados en 'combined_data.csv'.")
    
    except Exception as e:
     print(f"Error al guardar el archivo CSV combinado: {e}")


except Exception as e:
    print(f"Se produjo un error: {e}")
