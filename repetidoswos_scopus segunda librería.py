#pip install rapidfuzz
#pip install fuzzywuzzy python-Levenshtein

import pandas as pd
import spacy
import re
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS  # Stopwords en inglés
from rapidfuzz import fuzz, process

try:
    # Cargar modelo de spaCy en inglés (usa el modelo en_core_web_lg)
    nlp = spacy.load('en_core_web_lg')
    
    # Función para preprocesar los títulos
    # Ejemplo:
    # title = "Análisis de la percepción (2023) sobre el clima, en ciudades latinoamericanas!"
    # print(preprocess_title(title))
    # Salida: analisis percepcion clima ciudades latinoamericanas
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
    scopus_file_path = 'G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\datascopus.csv'
    wos_file_path = 'G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\dataexcel.xls'

    try:
        # Leer los datos del archivo CSV de Scopus
        scopus_df = pd.read_csv(scopus_file_path)
        
        # Leer los datos del archivo Excel de Web of Science
        wos_df = pd.read_excel(wos_file_path)
        wos_df['Authors'] = wos_df['Authors'].str.replace(',', '')
        wos_df['Author(s) ID'] = wos_df['Authors']
        wos_df['Source'] = 'Scopus'
        wos_df['Publication Stage'] = 'Final'
       

    except Exception as e:
        print(f"Error al cargar los archivos: {e}")
        raise

    # Guardar conteos originales para la estadística
    original_scopus_count = len(scopus_df)
    original_wos_count = len(wos_df)

    # Preprocesar los títulos en ambos dataframes
    scopus_df['processed_title'] = scopus_df['Title'].apply(preprocess_title)
    wos_df['processed_title'] = wos_df['Article Title'].apply(preprocess_title)
    
    # Valores iniciales
    print("**Valores iniciales:**")
    print(f"Total de artículos en Scopus: {original_scopus_count}")
    print(f"Total de artículos en Web of Science: {original_wos_count}\n")
    
    # Contar los duplicados internos antes de eliminarlos
    scopus_duplicates_count = scopus_df.duplicated(subset=['processed_title']).sum()
    wos_duplicates_count = wos_df.duplicated(subset=['processed_title']).sum()
    print(f"Duplicados en Scopus antes de eliminar: {scopus_duplicates_count}")
    print(f"Duplicados en Web of Science antes de eliminar: {wos_duplicates_count}\n")

    # Eliminar duplicados dentro de cada dataframe basado en los títulos procesados
    scopus_df = scopus_df.drop_duplicates(subset=['processed_title'])
    wos_df = wos_df.drop_duplicates(subset=['processed_title'])
    
    # Normalizar la columna DOI en ambos dataframes
    scopus_df['DOI'] = scopus_df['DOI'].fillna('').str.lower().str.strip()
    wos_df['DOI'] = wos_df['DOI'].fillna('').str.lower().str.strip()

    # (1) Detección de duplicados por DOI
    doi_matches = []
    scopus_dois = set(scopus_df['DOI'].values)  # para búsquedas rápidas

    for idx, wos_row in wos_df.iterrows():
        wos_doi = wos_row['DOI']
        if wos_doi and wos_doi in scopus_dois:
            # DOI coincide => duplicado seguro
            doi_matches.append(wos_row['processed_title'])

    # (2) Fuzzy matching para títulos
    threshold_fuzzy = 90  # umbral de similitud
    similar_titles = []

    # Convertir los títulos de Scopus en lista para fuzzy matching
    scopus_titles_list = scopus_df['processed_title'].tolist()

    for idx, wos_row in wos_df.iterrows():
        wos_title = wos_row['processed_title']
        wos_doi = wos_row['DOI']

        # Si ya se detectó duplicado por DOI, saltamos
        if wos_title in doi_matches:
            continue

        if wos_doi and wos_doi in scopus_dois:
            doi_matches.append(wos_title)
            continue
        
        # Fuzzy matching: retorna (best_match, score, match_index)
        best_match, score, _ = process.extractOne(
            wos_title,
            scopus_titles_list,
            scorer=fuzz.token_sort_ratio
        )
        
        if score > threshold_fuzzy:
            similar_titles.append(wos_title)

    # Combinar los duplicados encontrados
    all_duplicates = set(doi_matches + similar_titles)
    print(f"Duplicados detectados por DOI: {len(doi_matches)}")
    print(f"Duplicados detectados por fuzzy: {len(similar_titles)}")
    print(f"n total hay {len(scopus_df) + len(wos_df)} artículos, En total hay {len(all_duplicates)} artículos repetidos.\n")

    # --- 5) Guardar los títulos repetidos en un archivo CSV ---
    output_file_path = "G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopus_repeatedstitles.csv"
    repeated_titles_df = pd.DataFrame(list(all_duplicates), columns=['Título Repetido'])
    
    try:
        repeated_titles_df.to_csv(output_file_path, index=False)
        print("Los títulos repetidos han sido guardados en 'wos_scopus_repeatedstitles.csv'.\n")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
    #wos_df['Ciudad'] = np.nan 
    # --- 6) Eliminar los títulos repetidos en wos_df ---
    wos_non_repeated = wos_df[~wos_df['processed_title'].isin(all_duplicates)]
    
    # Renombrar columnas de WoS para que coincidan con Scopus
    df_wos_renombrado = wos_non_repeated.rename(columns={
               'Publication Stage' :'Publication Stage',

        'Source': 'Source',
        'UT (Unique WOS ID)': 'EID',
        'Author(s) ID': 'Author(s) ID',
        'Document Type': 'Document Type',
        'Language': 'Language of Original Document',
        'Author Keywords': 'Author Keywords',
        'Keywords Plus': 'Index Keywords',
            'Abstract': 'Abstract',
        'DOI': 'DOI',
        'Author Full Names': 'Author full names',
        'Authors':'Authors',
        'Cited Reference Count': 'Cited by',
        'Publication Year': 'Year',
        #'UT (Unique WOS ID)': '',
        'Source Title': 'Source title',
        'Article Title': 'Title',
        'Addresses': 'Authors with affiliations',
        'Open Access Designations': 'Open Access',
        'ISSN': 'ISSN',
        'Publisher': 'Publisher',
        'DOI Link': 'Link'
    })

    # Seleccionar solo las columnas necesarias en Web of Science
    necessary_columns = [
       'Publication Stage', 'Authors','Author(s) ID','Source','EID', 'Document Type', 'Language of Original Document', 'Author Keywords', 
        'Index Keywords', 'Abstract', 'DOI', 'Cited by', 'Year', 'Source title', 
        'Title', 'Affiliations', 'ISSN', 'Publisher', 'Link', 'Open Access', 'Author full names',
        'Scopus_SubjectArea', 'Authors with affiliations', 'processed_title'
    ]
    final_cols = [col for col in necessary_columns if col in df_wos_renombrado.columns]
    df_wos_renombrado = df_wos_renombrado[final_cols]
    
    # --- 8) Combinar datos ---
    # Concatenar los datos de Scopus y WoS (ya procesados)
    combined_df = pd.concat([scopus_df, df_wos_renombrado], ignore_index=True)
    # Filtrar por años (2014 a 2024)
    filtro = (combined_df['Year'] >= 2014) & (combined_df['Year'] <= 2024)
    combined_df = combined_df.loc[filtro]
    
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

    # Aplicar la función a la columna 'Authors'
    combined_df['Authors'] = combined_df['Authors'].apply(process_authors)
    
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
        
    combined_df['Source title'] = combined_df['Source title'].apply(process_source_title)
    
    # Validar y rellenar valores nulos entre columnas de afiliaciones
    def fill_missing_values(row):
        affiliations = row['Affiliations']
        authors_with_affiliations = row['Authors with affiliations']
        if (pd.isna(affiliations) or affiliations.strip() == "") and \
           (pd.isna(authors_with_affiliations) or authors_with_affiliations.strip() == ""):
            return pd.Series([affiliations, authors_with_affiliations], index=['Affiliations', 'Authors with affiliations'])
        if pd.isna(affiliations) or affiliations.strip() == "":
            affiliations = authors_with_affiliations
        if pd.isna(authors_with_affiliations) or authors_with_affiliations.strip() == "":
            authors_with_affiliations = affiliations
        return pd.Series([affiliations, authors_with_affiliations], index=['Affiliations', 'Authors with affiliations'])
    
    combined_df[['Affiliations', 'Authors with affiliations']] = combined_df.apply(fill_missing_values, axis=1)
    
    # Función para normalizar países
    def normalize_country(country):
        country = re.sub(r'(?i)\b(usa|u\.s\.a\.|united states of america|united states)\b', 'United States', country)
        country = re.sub(r'(?i)\bpeoples r china\b', 'China', country)
        country = re.sub(r'(?i)\brussian federation\b', 'Russia', country)
        country = re.sub(r'(?i)\bengland\b', 'United Kingdom', country)
        country = re.sub(r'(?i)\bir\b', 'Iran', country)
        country = re.sub(r'(?i)\bviet nam\b', 'Vietnam', country)
        country = re.sub(r'\s+', ' ', country).strip()
        return country
    
    # Función para procesar cada registro (afiliaciones)
    def process_record(record):
        if pd.isna(record):
            return record
        record = re.sub(r'\[(.*?)\]', lambda m: m.group(0).replace(',', ''), record)  
        fragments = record.split(';')
        processed_fragments = []
        for fragment in fragments:
            fragment = fragment.strip()
            if ',' in fragment:
                parts = fragment.rsplit(',', 1)
                main_info = parts[0].strip()
                country_info = parts[1].strip()
                country_info = re.sub(r'[0-9]+', '', country_info)
                country_info = re.sub(r'\b[A-Z]{1,2}\b', '', country_info)
                country_info = country_info.strip()
                normalized_country = normalize_country(country_info)
                processed_fragments.append(f"{main_info}, {normalized_country}")
            else:
                processed_fragments.append(fragment)
        return '; '.join(processed_fragments)
    
    # Aplicar la función a las columnas seleccionadas
    columns_to_process = ['Affiliations', 'Authors with affiliations']
    for column in columns_to_process:
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].apply(process_record)

    # --------------------------------------------------------------
    # Bloque para calcular y mostrar las estadísticas de salida
    # --------------------------------------------------------------
    total_loaded = original_scopus_count + original_wos_count
    omitted_papers = 0  # No se omitieron artículos por tipo de documento
    after_omission_total = total_loaded

    scopus_unique_count = len(scopus_df)
    # En wos_df, ya se eliminaron duplicados internos; 
    # Los duplicados inter-base se eliminaron al filtrar: 
    removed_wos = original_wos_count - len(wos_non_repeated)
    removed_scopus = original_scopus_count - scopus_unique_count

    duplicated_papers_found = len(all_duplicates)
    
    final_total = len(combined_df)
    final_wos_count = len(df_wos_renombrado)
    final_scopus_count = scopus_unique_count

    percentage_wos_loaded = (original_wos_count / total_loaded) * 100
    percentage_scopus_loaded = (original_scopus_count / total_loaded) * 100

    final_wos_percentage = (final_wos_count / final_total) * 100
    final_scopus_percentage = (final_scopus_count / final_total) * 100

    removed_wos_percentage = (removed_wos / original_wos_count) * 100 if original_wos_count != 0 else 0
    removed_scopus_percentage = (removed_scopus / original_scopus_count) * 100 if original_scopus_count != 0 else 0
    duplicated_percentage = (duplicated_papers_found / total_loaded) * 100

    print("\n***** Original Data *****")
    print(f"Loaded papers: {total_loaded}")
    print(f"Omitted papers by document type: {omitted_papers} ({0.0}%)")
    print(f"Total papers after omitted papers removed: {after_omission_total}")
    print(f"Loaded papers from WoS: {original_wos_count} ({percentage_wos_loaded:.1f}%)")
    print(f"Loaded papers from Scopus: {original_scopus_count} ({percentage_scopus_loaded:.1f}%)\n")

    print("Duplicated removal results:")
    print(f"Duplicated papers found: {duplicated_papers_found} ({duplicated_percentage:.1f}%)")
    print(f"Removed duplicated papers from WoS: {removed_wos} ({removed_wos_percentage:.1f}%)")
    print(f"Removed duplicated papers from Scopus: {removed_scopus} ({removed_scopus_percentage:.1f}%)")
    print(f"Total papers after duplicates removal: {final_total}")
    print(f"Papers from WoS: {final_wos_count} ({final_wos_percentage:.1f}%)")
    print(f"Papers from Scopus: {final_scopus_count} ({final_scopus_percentage:.1f}%)\n")

    print("Statics after duplication removal filter:")
    print(f"        WoS: {final_wos_count} ({final_wos_percentage:.1f}%)")
    print(f"        Scopus: {final_scopus_count} ({final_scopus_percentage:.1f}%)\n")
    
    # Generar gráfico de barras horizontal con la distribución final
    sources = ['WoS', 'Scopus']
    final_percentages = [final_wos_percentage, final_scopus_percentage]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sources, final_percentages)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Distribución de artículos únicos por fuente")
    plt.tight_layout()
    plt.show()
    print(combined_df.dtypes)
    for col in ["Volume", "Page count", "PubMed ID"]:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce') \
                        .astype("Int64")

    # 2. Eliminar la columna 'processed_title'
    combined_df.drop(columns="processed_title", inplace=True)




    # --------------------------------------------------------------
    # Guardar el DataFrame combinado en un archivo CSV
    combined_output_file_path = "G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopuslibrería.csv"
    try:
        combined_df.to_csv(combined_output_file_path, index=False)
        print("**Resultados finales:**")
        print(f"Total de artículos únicos combinados: {final_total}")
        print("Los datos combinados han sido guardados en 'wos_scopuslibrería.csv'.")
    except Exception as e:
        print(f"Error al guardar el archivo CSV combinado: {e}")

except Exception as e:
    print(f"Se produjo un error: {e}")
