#pip install rapidfuzz
#pip install fuzzywuzzy python-Levenshtein
#linea 247
import pandas as pd
import spacy
import re
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS  # Stopwords en inglés
from rapidfuzz import fuzz, process

#VARIABLES 
YEAR_START = 2014
YEAR_FINAL = 2024
UMBRAL = 90
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
        # 5. Lematización (opcional)
        doc = nlp(title)
        lemmas = [
            
            token.lemma_ if token.lemma_ != "-PRON-" else token.text
            for token in doc
                    # Filtra nuevamente stop-words y signos por seguridad
            if not (token.is_stop or token.is_punct or token.is_space)          ]
        
        title = " ".join(lemmas)
        
        return title
    def normalize_document_type(doc_type):
        """
        Normaliza los tipos de documento en Web of Science y Scopus, manejando casos con ';'.
        """
        if pd.isna(doc_type):
            return np.nan  # Maneja valores nulos

        

        # Reemplazar todos los términos que corresponden a "Conference Paper"
        doc_type = doc_type.replace('Proceedings Paper', 'Conference paper')

        # Normalizar tipos de documento de conferencias
        if 'conference paper' in doc_type:
            doc_type = 'Conference Paper'

  

        # Filtrar el exceso de espacios y mantener el formato de ";"
        doc_type = re.sub(r'\s*;\s*', '; ', doc_type).strip()

        return doc_type
    # Cargar los dato

#scopus_file_path = 'G:\\Mi unidad\\2025\\Master Italo Palacios\\articulo\\datascopus.csv'

    scopus_file_path = r"G:\Mi unidad\2025\Master  FRANCISCO MARCELO ALVARADO PORRAS\data\datascopus.csv"
    scimago_ruta = r"G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\scimago_unificado.csv"

    wos_file_path = r'G:\Mi unidad\2025\Master  FRANCISCO MARCELO ALVARADO PORRAS\data\datawos.xls'

    try:
        scimagodata = pd.read_csv(scimago_ruta, sep=";")
        # Leer los datos del archivo CSV de Scopus
        scopus_df = pd.read_csv(scopus_file_path)
        
        # Leer los datos del archivo Excel de Web of Science
        wos_df = pd.read_excel(wos_file_path)
        def clean_data(cat_str):
            # Si la celda está vacía, devuelve NaN o cadena vacía
            if pd.isna(cat_str):
                return np.nan          # o '' si prefieres dejarla vacía
            
            cat_str = str(cat_str)     # fuerza a cadena
            # 1) Elimina todo lo que esté entre paréntesis
            no_par = re.sub(r'\([^)]*\)', '', cat_str)
            # 2) Divide por “;”, recorta espacios y descarta vacíos
            parts = [p.strip() for p in no_par.split(';') if p.strip()]
            # 3) Vuelve a unir con “; ”
            return '; '.join(parts)
        scopus_df['Author full names'] = scopus_df['Author full names'].apply(clean_data)
        wos_df['Authors'] = wos_df['Authors'].str.replace(',', '')

        wos_df['Author(s) ID'] = wos_df['Authors']
        wos_df['Source'] = 'Web of science'
        wos_df['Publication Stage'] = 'Final'
        wos_df['Source Title'] = wos_df['Source Title'].str.replace('&', 'and', regex=False)
        wos_df['Document Type'] = wos_df['Document Type'].apply(normalize_document_type)

       

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
    threshold_fuzzy = UMBRAL  # umbral de similitud
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
            scorer=fuzz.WRatio
        )
        
        if score > threshold_fuzzy:
            similar_titles.append(wos_title)

    # Combinar los duplicados encontrados
    all_duplicates = set(doi_matches + similar_titles)
    print(f"Duplicados detectados por DOI: {len(doi_matches)}")
    print(f"Duplicados detectados por fuzzy: {len(similar_titles)}")
    print(f"n total hay {len(scopus_df) + len(wos_df)} artículos, En total hay {len(all_duplicates)} artículos repetidos.\n")
    # En WoS, todos los registros que quedan son exclusivos (0)
    wos_df["In_Both"] = 0
    wos_df["In_Both"] = wos_df["processed_title"].isin(all_duplicates).astype(int)
    scopus_df["In_Both"] = scopus_df["processed_title"].isin(all_duplicates).astype(int)
    # --- 5) Guardar los títulos repetidos en un archivo CSV ---
    output_file_path = r"G:\Mi unidad\2025\Master  FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopus_repeatedstitles.csv"
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
    # Filtrar por años (year star  a year final)
    filtro = (combined_df['Year'] >= YEAR_START) & (combined_df['Year'] <= YEAR_FINAL)
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
    #combined_df['Authors'] = combined_df['Authors'].apply(process_authors)
    #combined_df['Source title'] = combined_df['Source title'].str.replace(',', '')
    # Normalizar ISSN en df_main
    combined_df['ISSN'] = (
        combined_df['ISSN']
        .replace({'': pd.NA})
        .str.replace(r'[^0-9X]', '', regex=True)
        .str.upper()
    )
    # Normalizar ISSN en SCImago para manejar múltiples valores
    scimago_expanded = scimagodata.assign(Issn=scimagodata['Issn'].str.split(',')).explode('Issn')
    scimago_expanded['Issn'] = scimago_expanded['Issn'].str.strip()  # Eliminar espacios adicionales en ISSN
        # Mapa ISSN → Título canónico (el primero encontrado)
    scimago_map = scimago_expanded.groupby('Issn')['Title'].first().to_dict()

    
    # Lista global para almacenar títulos únicos de revistas
    # Función de asignación priorizando Scopus y luego fuzzy match
    def _safe_text(x):
        """Convierte a str solo si hay texto; si es NaN/None devuelve ''."""
        if isinstance(x, str):
            return x
        if pd.isna(x):
            return ""
        return str(x)

    def assign_canonical_title(row):
        # Lee campos de forma segura
        issn = _safe_text(row.get('ISSN', '')).strip()
        src  = _safe_text(row.get('Source', '')).strip().lower()
        orig = _safe_text(row.get('Source title', '')).strip()

        # 1) Si hay ISSN y existe en el catálogo SCImago y el registro NO viene de Scopus,
        #    usamos el título canónico de SCImago (limpiando paréntesis).
        if issn and (issn in scimago_map) and (src != 'scopus'):
            cand = _safe_text(scimago_map.get(issn, '')).strip()
            if cand:
                return re.sub(r'\([^)]*\)', '', cand).strip()

        # 2) Si NO hay ISSN, probamos fuzzy contra el catálogo (siempre que tengamos 'orig')
        if (not issn) and (src != 'scopus') and orig:
            best = process.extractOne(orig, list(scimago_map.values()), scorer=fuzz.token_sort_ratio)
            if best:
                best_title, score, _ = best
                if isinstance(score, (int, float)) and score > 90 and isinstance(best_title, str):
                    return re.sub(r'\([^)]*\)', '', best_title).strip()

        # 3) En cualquier otro caso, devolvemos el original (limpiando paréntesis si hay texto)
        return re.sub(r'\([^)]*\)', '', orig).strip() if orig else orig
    
    
    for col in ['ISSN', 'Source', 'Source title']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(object)  # evita conversión implícita a float
            combined_df[col] = combined_df[col].where(~combined_df[col].isna(), None)

    # (Opcional) Si quieres que 'Source' nunca sea nulo:
    combined_df['Source'] = combined_df['Source'].fillna('unknown')
    combined_df['Source title'] = combined_df.apply(assign_canonical_title, axis=1)
    combined_df["In_Both"] = combined_df["In_Both"].fillna(0).astype(int)


    #combined_df['Author full names'] = combined_df['Authors']
 
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
    def normalize_country(text):
        # aquí lanzas cada re.sub sobre todo el record
  
        text = re.sub(r'(?i)\b(?:usa|u\.s\.a\.|united states of america|united states)\b', 'United States', text)
        text = re.sub(r'(?i)\b(?:uk|u\.k\.|united kingdom)\b', 'United Kingdom', text)
        text = re.sub(r'(?i)\b(?:united arab emirates)\b', 'United Arab Emirates', text)
        text = re.sub(r'(?i)\brepublic of korea\b', 'South Korea', text)
        text = re.sub(r'(?i)\brepublic of korea\b', 'South Korea', text)
        text = re.sub(r'(?i)\bpeoples r china\b', 'China', text)
        text = re.sub(r'(?i)\brussian federation\b', 'Russia', text)
        text = re.sub(r'(?i)\bengland\b', 'United Kingdom', text)
      
        text = re.sub(r'(?i)\bScotland\b', 'United Kingdom', text)
        text = re.sub(r'(?i)\bwales\b', 'United Kingdom', text)
        text = re.sub(r'(?i)\bnorthern ireland\b', 'United Kingdom', text)
   
    
     
        text = re.sub(r'(?i)\bSt Martin\b', 'United Kingdom', text)
        # Un solo patrón para Viet Nam (insensible a espacios)
        text = re.sub(r'(?i)\bviet\s?nam\b', 'Vietnam', text)
        # Ivory Coast
        text = re.sub(r"(?i)\bCôte d'Ivoire\b", "Ivory Coast", text)
        text = re.sub(r"(?i)\bCote d'Ivoire\b", "Ivory Coast", text)
        text = re.sub(r"(?i)\bCote Ivoire\b",   "Ivory Coast", text)        
        text = re.sub(r"(?i)\bDominican Rep\b", "Dominican Republic", text)
        
        
        text = re.sub(r"(?i)\bTrinidad Tobago\b", "Trinidad and Tobago", text)
        text = re.sub(r"(?i)\bTimor Leste\b", "Timor-Leste", text)
        text = re.sub(r"(?i)\bSt Vincent\b", "Saint Vincent and the Grenadines", text)
        text = re.sub(r"(?i)\bGermany (Democratic Republic, DDR)\b", "Germany", text)
        text = re.sub(r"(?i)\bSao Tome & Prin\b", "Sao Tome and Principe", text)
        text = re.sub(r"(?i)\bSt Lucia\b", "Saint Lucia", text)
        
        text = re.sub(r"(?i)\bSt Kitts & Nevi\b", "Saint Kitts and Nevis", text)
        
        text = re.sub(r"(?i)\bPapua N Guinea\b", "Papua New Guinea", text)
        text = re.sub(r"(?i)\bGuinea Bissau\b", "Guinea-Bissau", text)
        text = re.sub(r"(?i)\bCent Afr Republ\b", "Central African Republic", text)
        
        text = re.sub(r"(?i)\bCape Verde\b", "Cabo Verde", text)
        text = re.sub(r"(?i)\bBrunei\b", "Brunei Darussalam", text)

        text = re.sub(r"(?i)\bNigeria\b", "Niger", text)

        text = re.sub(r"(?i)\bDEM REP CONGO\b", "Congo", text)
        text = re.sub(r"(?i)\bDemocratic Republic of the Congo\b", "Congo", text)
        text = re.sub(r"(?i)\bDominican Rep\b", "Dominican Republic", text)
        text = re.sub(r"(?i)\bTurkiye\b", "Turkey", text)
        text = re.sub(r"(?i)\bSt Martin\b", "Saint Martin", text)
        text = re.sub(r"(?i)\bSaint Martin\b", "Saint Martin", text)
        
        
        
        
        return text

    # —————————————————————————————
    # Nuevo process_record: solo limpia corchetes y normaliza el texto entero
    # —————————————————————————————
    def process_record(record):
        if pd.isna(record):
            return record
        # 1) quitar comas internas en corchetes
        record = re.sub(r'\[(.*?)\]', lambda m: m.group(0).replace(',', ''), record)
        # 2) normalizar cualquier mención de país dentro del texto
        record = normalize_country(record)
        return record

    # —————————————————————————————
    # Aplicación a las columnas
    # —————————————————————————————
    for col in ['Affiliations', 'Authors with affiliations']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(process_record)
    combined_df['Authors'] = combined_df['Author full names']
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
    combined_df["Authors"] = combined_df['Author full names']
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
# Tus datos reales:
    sources     = ['WoS', 'Scopus']
    kept        = [final_wos_count, final_scopus_count]
    removed     = [removed_wos, removed_scopus]

    # Total por fuente
    totals = np.array(kept) + np.array(removed)

    # ORDENAR DE MAYOR A MENOR
    order = np.argsort(totals)[::-1]

    sources  = [sources[i] for i in order]
    kept     = [kept[i]    for i in order]
    removed  = [removed[i] for i in order]
    totals   = np.array(kept) + np.array(removed)

    pct_kept    = np.array(kept) / totals * 100
    pct_removed = np.array(removed) / totals * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    bars_kept    = ax.barh(sources, kept, label='Kept')
    bars_removed = ax.barh(sources, removed, left=kept, label='Removed')

    # Etiquetas
    for i, (b1, b2) in enumerate(zip(bars_kept, bars_removed)):
        w1 = b1.get_width()
        c1 = 'white' if w1 > totals[i]*0.15 else 'black'
        ax.text(w1/2,
                b1.get_y()+b1.get_height()/2,
                f'{kept[i]}\n({pct_kept[i]:.1f}%)',
                va='center', ha='center', color=c1)

        w2 = b2.get_width()
        if w2 > 0:
            c2 = 'white' if w2 > totals[i]*0.15 else 'black'
            ax.text(kept[i] + w2/2,
                    b2.get_y()+b2.get_height()/2,
                    f'{removed[i]}\n({pct_removed[i]:.1f}%)',
                    va='center', ha='center', color=c2)

    # --- ESTO ASEGURA QUE EL MAYOR SIEMPRE ESTÉ ARRIBA ---
    ax.invert_yaxis()

    ax.set_title("Post-deduplication Distribution of Bibliometric Records\nfrom Scopus and Web of Science",
                weight='bold', pad=12)
    ax.set_xlabel("Number of Articles")
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    for col in ["Volume", "Page count", "PubMed ID"]:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce') \
                        .astype("Int64")
# Reemplazar vacíos o valores no numéricos por 0
    combined_df["Cited by"] = pd.to_numeric(combined_df["Cited by"], errors="coerce").fillna(0)

# Convertir a enteros
    combined_df["Cited by"] = combined_df["Cited by"].astype(int)
    # Reemplazar vacíos o nulos por "subscription"
    combined_df["Open Access"] = combined_df["Open Access"].fillna("subscription")
    # También cubrir casos de cadenas vacías " "
    combined_df["Open Access"] = combined_df["Open Access"].replace(r'^\s*$', "subscription", regex=True)
    # 2. Eliminar la columna 'processed_title'
    combined_df.drop(columns="processed_title", inplace=True)
    print("**Resultados finales:**")
    print(f"Total de artículos únicos combinados: {final_total}")


    # ——————————————————————————————————————————————————
    # Asegúrate de que las columnas de año estén en numérico/int
    # ——————————————————————————————————————————————————
    scopus_df['Year'] = pd.to_numeric(scopus_df['Year'], errors='coerce')
    wos_df['Year']    = pd.to_numeric(wos_df['Publication Year'], errors='coerce')

    # ——————————————————————————————————————————————————
    # Filtrar rangos de interés (2014–2024)
    # ——————————————————————————————————————————————————
    mask_sc = scopus_df['Year'].between(YEAR_START, YEAR_FINAL)
    mask_wo = wos_df['Year'].between(YEAR_START, YEAR_FINAL)
    # ——————————————————————————————————————————————————
    # Conteo de artículos por año
    # ——————————————————————————————————————————————————
    raw_scopus_yearly = scopus_df.loc[mask_sc].groupby('Year').size()
    raw_wos_yearly    = wos_df.loc[mask_wo].groupby('Year').size()

    # ——————————————————————————————————————————————————
    # Suma de citas por año
    # ——————————————————————————————————————————————————
    raw_scopus_cites = scopus_df.loc[mask_sc].groupby('Year')['Cited by'].sum()
    raw_wos_cites    = wos_df.loc[mask_wo].groupby('Year')['Cited Reference Count'].sum()

    # ——————————————————————————————————————————————————
    # Unir en DataFrames
    # ——————————————————————————————————————————————————
    raw_counts = pd.DataFrame({
        'WoS':    raw_wos_yearly,
        'Scopus': raw_scopus_yearly
    }).fillna(0).astype(int)
    raw_counts['Total Articles Raw'] = raw_counts.sum(axis=1)

    raw_citations = pd.DataFrame({
        'WoS Citations':    raw_wos_cites,
        'Scopus Citations': raw_scopus_cites
    }).fillna(0).astype(int)
    raw_citations['Total Citations Raw'] = raw_citations.sum(axis=1)

    # ——————————————————————————————————————————————————
    # Imprimir tablas por consola
    # ——————————————————————————————————————————————————
    print("=== Raw Article Counts by Year ===")
    print(raw_counts)
    print("\n=== Raw Citation Counts by Year ===")
    print(raw_citations)

    # ——————————————————————————————————————————————————
    # Gráfico 1: evolución de artículos crudos
    # ——————————————————————————————————————————————————
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(raw_counts.index, raw_counts['WoS'],              marker='o', label='WoS Articles')
    ax.plot(raw_counts.index, raw_counts['Scopus'],           marker='s', label='Scopus Articles')
    ax.plot(raw_counts.index, raw_counts['Total Articles Raw'], marker='^', label='Total Articles')
    ax.set_title("Annual evolution of articles (RAW data before deduplication)",
                weight='bold', pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of articles")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ——————————————————————————————————————————————————
    # Gráfico 2: evolución de citas crudas
    # ——————————————————————————————————————————————————
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(raw_citations.index, raw_citations['WoS Citations'],       marker='o', label='WoS Citations')
    ax.plot(raw_citations.index, raw_citations['Scopus Citations'],    marker='s', label='Scopus Citations')
    ax.plot(raw_citations.index, raw_citations['Total Citations Raw'], marker='^', label='Total Citations')
    ax.set_title("Annual evolution of citations (RAW data before deduplication)",
                weight='bold', pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of citations")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    # 1. Convertir Year a numérico
    combined_df['Year'] = pd.to_numeric(combined_df['Year'], errors='coerce')



    # Configuración para artículo científico
    plt.style.use('default')  # Estilo base limpio
    plt.rcParams['font.family'] = 'serif'  # Fuente serif para publicación
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

    # 1. Convertir Year a numérico
    #combined_df['Year'] = pd.to_numeric(combined_df['Year'], errors='coerce')

    # 2. Crear una copia del DataFrame con las columnas necesarias
    df_to_explode = combined_df[['Year', 'Document Type']].copy()

    # 3. Dividir y limpiar los tipos de documento
    df_to_explode['Document Type'] = df_to_explode['Document Type'].str.split(';')
    df_to_explode['Document Type'] = df_to_explode['Document Type'].apply(
        lambda x: [item.strip() for item in x if item.strip()] if isinstance(x, list) else [x]
    )

    # 4. Explotar el DataFrame (crear una fila por cada tipo de documento)
    exploded_df = df_to_explode.explode('Document Type')

    # 5. Filtrar filas con Year válido (eliminar filas con valores nulos en Year)
    exploded_df = exploded_df.dropna(subset=['Year'])

    # 6. Agrupar por año y tipo de documento, y contar
    yearly_document_counts = exploded_df.groupby(['Year', 'Document Type']).size().unstack(fill_value=0)

    # 7. Crear el gráfico con estilo científico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Paleta de colores para artículo científico (accesible y profesional)
    scientific_colors = [
        '#4C72B0',  # Azul
        '#DD8452',  # Naranja
        '#55A868',  # Verde
        '#C44E52',  # Rojo
        '#8172B3',  # Púrpura
        '#937860',  # Marrón
        '#DA8BC3',  # Rosa
        '#8C8C8C',  # Gris
        '#CCB974',  # Oro
        '#64B5CD'   # Turquesa
    ]

    # Asegurar que tenemos suficientes colores
    if len(yearly_document_counts.columns) > len(scientific_colors):
        # Generar colores adicionales si son necesarios
        cmap = plt.cm.get_cmap('Set3', len(yearly_document_counts.columns))
        scientific_colors = [cmap(i) for i in range(len(yearly_document_counts.columns))]

    # Crear el gráfico de barras apiladas
    yearly_document_counts.plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        color=scientific_colors[:len(yearly_document_counts.columns)]
    )

    # Personalizar el gráfico para artículo científico
    ax.set_title("Distribution of Document Types by Year", fontweight='bold', pad=15)
    ax.set_xlabel("Year", fontweight='bold')
    ax.set_ylabel("Number of Documents", fontweight='bold')

    # Colocar la leyenda dentro del gráfico
    ax.legend(
        title="Document Type", 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1),
        title_fontsize=11,
        frameon=True,
        edgecolor='black',
        fancybox=False
    )

    # Ajustar las etiquetas del eje X
    plt.xticks(rotation=45, ha='right')

    # Ajustar el diseño
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


    # Mostrar tabla de conteos
    print("Tabla de distribución de tipos de documento por año:")
    print(yearly_document_counts)
        # --------------------------------------------------------------
    # Guardar el DataFrame combinado en un archivo CSV
    combined_output_file_path = r"G:\Mi unidad\2025\Master  FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopus.csv"
    try:
        combined_df.to_csv(combined_output_file_path, index=False)
       
        print("Los datos combinados han sido guardados en 'wos_scopuslibrería.csv'.")
    except Exception as e:
        print(f"Error al guardar el archivo CSV combinado: {e}")

except Exception as e:
    print(f"Se produjo un error: {e}")
