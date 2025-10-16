import pandas as pd

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")


df = pd.read_csv(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replacelematizar.csv")
   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [

   "adult",
    "male",
    "female",
    "human experiment",
    "major clinical study",
    "nurse",
    "telemedicine",
    "telehealth",
    "medical informatic",
    "primary medical care",
    "clinical article",
    "hospital",
    "physician",
    "nursing student",
    "nurse administrator",
    "attitude of health personnel",
    "patient care",
    "sar cov-2",
    "construction industry",
    "commerce",
    "sale",
    "smart city",
    "supply chain",
    "agriculture",
    "biochemistry",
    "biotechnology",
    "chemical process",
    "civil engineering",
    "geology",
    "pharmacy",
    "physics",
    "toxicology",
    "environmental chemistry",
    "epidemiology",
    "drug therapy",
    "virus",
    "medical research",
    "medicine",
    "clinical practice",
    "telenursing",
    "health care access",
    "health care planning",
    "health data",
    "health insurance",
    "hospital patient",
    "pediatric",
    "midwife",
    "nurse 's role",
    "population health",
    "public hospital",
    "state medicine",
    "therapy",
    "medical care",
    "newborn",
    "oil and gas",
    "manufacturing",
    "industrial engineering",
    "industrial management",
    "industry professional",
    "industrial research",
    "gas industry",
    "vehicle",
    "robotic",
    "robot",
    "philosophical aspect",
    "spatiotemporal analysis",
    "genetic transcription",
    "biotechnology",
    "agriculture",
    "tourism",
    "veterinary",
    "epidemic",
    "epidemiology",
    "occupational health",
    "primary care",
    "health-oriented leadership",
        # Países, regiones o gentilicios
    "england",
    "finland",
    "philippine",
    "nigeria",
    "south africa",
    "south korea",
    "new south wale",
    "ontario",
    "united kingdom",
    "european union",
    "developing country",
    "local government",
    "municipality",
    "municipal administration",
    "public organization",
    "public sector",
    "national health service",
    "world health organization",
    
    # Plataformas o redes sociales
    "facebook",
    "twitter",
    "instagram",
    "google",
    
    # Entidades o siglas genéricas
    "hrm",
    "cio",
    "chief information officer",
    "king salman",   # ruido institucional
    "boston matrix",
    "crm",           # gestión comercial
    "stem",
    "stem science technology engineering and mathematic",
    "pressung",      # error OCR común en datasets
    "moocs"          # puede mantenerse si no es foco
        

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
df['Index Keywords'] = eliminar_palabras_clave(df['Index Keywords'])
#df['bothKeywords'] =  eliminar_palabras_clave(df['bothKeywords'])
df['Author Keywords'] = eliminar_palabras_clave(df['Author Keywords'])
# Guardar el DataFrame filtrado en un nuevo archivo CSV

# Construir listado global normalizado
all_terms = pd.concat([
    df['Index Keywords'].dropna(),
    df['Author Keywords'].dropna()
]).str.split(';').explode().str.strip()

# Frecuencia sobre versión normalizada
freq = all_terms.str.lower().value_counts()
keep_norm = set(freq[freq > 1].index)

def filter_unique(cell):
    if not isinstance(cell, str):
        return ""
    terms = [t.strip() for t in cell.split(';') if t.strip()]
    filtered = [t for t in terms if t.lower() in keep_norm]
    return '; '.join(filtered)
#df['Index Keywords'] = df['Index Keywords'].apply(filter_unique)
#df['Author Keywords'] = df['Author Keywords'].apply(filter_unique)
# Guardar el DataFrame filtrado en un nuevo archivo CSV
#df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)

df.to_csv(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replacelematizar2.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
