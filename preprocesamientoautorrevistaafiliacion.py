import pandas as pd
import re
import unicodedata

# Ruta del archivo combinado
combined_output_file_path = "G:\\Mi unidad\\2024\\Msc. Alberto León Batallas\\nuevoarticulo\\artículo final\\data\\wos_scopus.csv"

# Cargar el archivo CSV
data = pd.read_csv(combined_output_file_path)

# Validar que la columna 'Authors' existe
if 'Authors' not in data.columns:
    raise ValueError("La columna 'Authors' no existe en el archivo.")

# Función para procesar la columna 'Authors'
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
data['Authors'] = data['Authors'].apply(process_authors)

data['Author full names'] = data['Authors']
# Función para procesar la columna 'Source title'
def process_source_title(title):
    if not isinstance(title, str):
        return ""

    # Eliminar texto entre paréntesis, incluyendo los paréntesis
    title = re.sub(r'\([^)]*\)', '', title)

    # Reemplazar guiones por espacios
    title = title.replace('-', ' ')

    # Eliminar caracteres redundantes o múltiples espacios
    title = re.sub(r'\s+', ' ', title).strip()

    # Normalizar texto (remover acentos)
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Mantener acrónimos en mayúsculas, pero capitalizar títulos normales
    if title.isupper():
        return title
    else:
        # Capitalizar cada palabra (formato título)
        return " ".join([word.capitalize() for word in title.split()])
    
data['Source title'] = data['Source title'].apply(process_source_title)


# este de aquí valida las columnas que estén nulas
# Función para rellenar valores entre columnas
def fill_missing_values(row):
    affiliations = row['Affiliations']
    authors_with_affiliations = row['Authors with affiliations']
    
    # Verificar si ambas están vacías
    if (pd.isna(affiliations) or affiliations.strip() == "") and \
       (pd.isna(authors_with_affiliations) or authors_with_affiliations.strip() == ""):
        return pd.Series([affiliations, authors_with_affiliations], index=['Affiliations', 'Authors with affiliations'])
    
    # Si 'Affiliations' está vacío, usar 'Authors with affiliations'
    if pd.isna(affiliations) or affiliations.strip() == "":
        affiliations = authors_with_affiliations
    
    # Si 'Authors with affiliations' está vacío, usar 'Affiliations'
    if pd.isna(authors_with_affiliations) or authors_with_affiliations.strip() == "":
        authors_with_affiliations = affiliations
    
    # Retornar ambos valores actualizados
    return pd.Series([affiliations, authors_with_affiliations], index=['Affiliations', 'Authors with affiliations'])

# Aplicar la función a cada fila
data[['Affiliations', 'Authors with affiliations']] = data.apply(fill_missing_values, axis=1)


#esto de aquí  valida que este los "[]" lo separa es más los valores que son de que son de wos la hace y contabiliza 
# Función para procesar cada fila de afiliaciones y autores
def process_affiliations_and_authors(row):
    affiliations = row['Affiliations']
    authors_with_affiliations = row['Authors with affiliations']
    
    # Verificar si ambas columnas están vacías
    if pd.isna(affiliations) and pd.isna(authors_with_affiliations):
        return ""
    
    # Determinar cuál columna utilizar como base
    column_to_use = affiliations if pd.notna(affiliations) and affiliations.strip() else authors_with_affiliations
    
    # Eliminar comas dentro de los corchetes []
    column_to_use = re.sub(r'\[(.*?)\]', lambda m: m.group(0).replace(',', ''), column_to_use)
    
    # Separar la información de autores y afiliaciones
    pattern = r'\[(.+?)\]\s*(.+?)($|; )'  # Captura autores en [] y sus respectivas afiliaciones
    matches = re.findall(pattern, column_to_use)
    
    if not matches:
        return column_to_use  # Si no cumple el patrón, devolver la columna sin cambios
    
    combined_info = []
    for authors_part, affiliation_part, _ in matches:
        # Separar autores por ";"
        authors_list = [author.strip() for author in authors_part.split(';') if author.strip()]
        
        # Si solo hay un autor en los corchetes
        if len(authors_list) == 1:
            combined_info.append(f"{authors_list[0]}, {affiliation_part.strip()}")
        else:
            # Si hay varios autores, emparejarlos con la misma afiliación
            for author in authors_list:
                combined_info.append(f"{author}, {affiliation_part.strip()}")
    
    # Unir las combinaciones con "; "
    return "; ".join(combined_info)

# Aplicar la función a las columnas

data['Affiliations'] = data['Affiliations'].apply(lambda x: process_affiliations_and_authors({'Affiliations': x, 'Authors with affiliations': None}))
data['Authors with affiliations'] = data['Authors with affiliations'].apply(lambda x: process_affiliations_and_authors({'Affiliations': None, 'Authors with affiliations': x}))




# Función para normalizar países
def normalize_country(country):
    country = re.sub(r'(?i)\b(usa|u\.s\.a\.|united states of america|united states)\b', 'United States', country)
    country = re.sub(r'(?i)\bpeoples r china\b', 'China', country)
    country = re.sub(r'(?i)\brussian federation\b', 'Russia', country)
    country = re.sub(r'(?i)\bengland\b', 'United Kingdom', country)
    country = re.sub(r'(?i)\bir\b', 'Iran', country)
    country = re.sub(r'(?i)\bviet nam\b', 'Vietnam', country)
    country = re.sub(r'\s+', ' ', country).strip()  # Eliminar espacios redundantes
    return country

# Función para procesar cada registro
def process_record(record):
    if pd.isna(record):
        return record  # Si el registro está vacío, devolverlo tal como está

    # Dividir el registro por ";"
    fragments = record.split(';')
    processed_fragments = []

    for fragment in fragments:
        fragment = fragment.strip()  # Eliminar espacios al inicio y al final

        if ',' in fragment:
            # Dividir por la última coma
            parts = fragment.rsplit(',', 1)
            main_info = parts[0].strip()  # Información principal
            country_info = parts[1].strip()  # Supuesto país
            
            # Extraer solo el país eliminando números y códigos redundantes
            country_info = re.sub(r'[0-9]+', '', country_info)  # Eliminar números
            country_info = re.sub(r'\b[A-Z]{1,2}\b', '', country_info)  # Eliminar códigos como "CA" o "IL"
            country_info = country_info.strip()  # Quitar espacios sobrantes
            
            # Normalizar el país
            normalized_country = normalize_country(country_info)
            processed_fragments.append(f"{main_info}, {normalized_country}")
        else:
            # Si no hay coma en el fragmento, mantenerlo sin cambios
            processed_fragments.append(fragment)

    # Reconstruir el registro con los fragmentos procesados
    return '; '.join(processed_fragments)

# Seleccionar columnas a procesar
columns_to_process = ['Affiliations','Authors with affiliations']


# Aplicar la función a las columnas seleccionadas
for column in columns_to_process:
    if column in data.columns:
        data[column] = data[column].apply(process_record)


# Guardar el resultado en un nuevo archivo CSV
output_file_path ="G:\\Mi unidad\\2024\\Msc. Alberto León Batallas\\nuevoarticulo\\artículo final\\data\\procesado_data.csv"
data.to_csv(output_file_path, index=False)

print("El procesamiento de la columna 'Authors' se ha completado y los datos se han guardado en 'procesado_dataafiliaciones.csv'.")
