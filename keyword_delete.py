import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

KW_COLS     = ["Author Keywords", "Index Keywords"]

#df = pd.read_csv(r"C:\Users\INVESTIGACION 47\Downloads\datawos_scopusreplace.csv")
   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [

    # ===== RUÍDO GENERAL =====
    "article", "research", "study", "model", "models", "system", "systems",
    "framework", "methodology", "methods", "approach", "analysis",
    "tool", "tools", "variables",

    # ===== DEMOGRÁFICOS (NO FOCO) =====
    "human", "humans", "male", "female", "adult", "aged", "young adult",
    "middle aged", "adolescent",

    # ===== PAÍSES / REGIONES =====
    "china", "united states", "australia", "saudi arabia", "indonesia",

    # ===== SALUD / CLÍNICO =====
    "mental health", "anxiety", "depression", "stress",
    "medical education", "nursing education", "medical student",
    "clinical article", "major clinical study",

    # ===== BIBLIOMÉTRICOS PUROS =====
    "bibliometric analysis", "bibliometrics",
    "systematic review", "systematic literature review",
    "meta-analysis", "metaanalysis",

    # ===== COVID =====
    "covid-19", "coronavirus disease 2019", "pandemic", "pandemics", "sars-cov-2",

    # ===== OTROS FUERA DE FOCO =====
    "accounting", "chemistry", "music", "physical education",
    "jobs", "employment", "job market",

    # ===== BASURA =====
    "0", "'current", "bothkeywords",
        # ===== GENERICAS =====
    "education",
    "technology",
    "science",
    "knowledge",
    "model",
    "models",
    "system",
    "systems",
    "framework",
    "design",
    "strategies",
    "challenges",
    "impact",
    "quality",
    "success",
    "skills",
    "future",
    "information",
    "language",
    "support",
    "tools",
    "process",
    "program",
    "work",
    "management",
    "organization",
    "media",
    "context",
    "variables",
    "environment",
    "software",
    "services",
    "approaches",
    "practices",
    "resources",
    "principles",
    "issues",
    "concept",
    "concepts",
    "dimensions",
    "factors",
    "goals",
    "history",
    "number",
    "time",
    "power",
    "identity",
    "culture",
    "values",

    # ===== METODOLOGICAS =====
    "pls-sem",
    "sem",
    "regression",
    "regression analysis",
    "structural equation modeling",
    "structural equation models",
    "fit indexes",
    "factor analysis",
    "cluster analysis",
    "network analysis",
    "thematic analysis",
    "qualitative research",
    "qualitative analysis",
    "quantitative analysis",
    "mixed methods",
    "case study",
    "case studies",
    "systematic review",
    "systematic literature review",
    "scoping review",
    "meta-analysis",
    "metaanalysis",
    "validation",
    "validity",
    "reliability",
    "survey",
    "questionnaire",
    "questionnaire survey",
    "interview",
    "interviews",
    "tests",
    "statistics",

    # ===== EDITORIALES =====
    "article",
    "review",
    "bibliometric",
    "bibliometrics",
    "bibliometric analysis",
    "prisma",
    "research",
    "research work",
    "academic research",
    "scientific research",

    # ===== HUMANOS / BIOLOGICOS =====
    "human",
    "humans",
    "male",
    "female",
    "adult",
    "children",
    "youth",
    "adolescents",
    "mental health",
    "depression",
    "stress",
    "anxiety",

    # ===== PAISES =====
    "china",
    "saudi arabia",
    "jordan",
    "malaysia",
    "south africa",
    "united states",
    "united kingdom",
    "germany",
    "spain",
    "india",
    "peru",
    "colombia",
    "ecuador",
    "indonesia",
    "philippines",
    "japan",
    "kazakhstan",
    "lebanon",
    "oman",
    "pakistan",
    "serbia",
    "thailand",
    "uae",
    "vietnam",
    "australia",
    "bangladesh",
    "nigeria",
    "latin america",
    "sub-saharan africa",

    # ===== NIVELES EDUCATIVOS NO RELEVANTES =====
    "k-12",
    "secondary education",
    "school",
    "schools",
    "primary education",

    # ===== DISCIPLINAS NO RELACIONADAS =====
    "chemistry",
    "biology",
    "pharmacy",
    "law",
    "mechanical engineering",
    "architecture",
    "music education",
    "physical education",
    "marketing education",
    "accounting education",

    # ===== TERMINOS TECNICOS DE IA (NO EDUCATIVOS) =====
    "algorithm",
    "algorithms",
    "optimization",
    "feature selection",
    "feature extraction",
    "contrastive learning",
    "federated learning",
    "adversarial machine learning",
    "generative adversarial networks",
    "ontology",
    "numerical model",
    "real-time systems",
    "precision",
    "classification",
    "decision tree",
    "decision trees",
    "random forest",
    "support vector regression",

    # ===== RUIDO TEXTUAL =====
    "'current",
    "al",
    "deep",
    "machine",
    "text",
    "fit",
    "ease",
    "self"

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
print("Antes (recuento únicos):")
for c in KW_COLS:
    if c in df.columns:
        nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
        print(f"  {c}: {nuniq}")
# Aplicar la función a las columnas "Index Keywords" y "Author Keywords"
#df['Index Keywords'] = eliminar_palabras_clave(df['Index Keywords'])
df['bothKeywords'] =  eliminar_palabras_clave(df['bothKeywords'])
#df['Author Keywords'] = eliminar_palabras_clave(df['Author Keywords'])
# Guardar el DataFrame filtrado en un nuevo archivo CSV
print("\nDespués (recuento únicos):")
for c in KW_COLS:
    if c in df.columns:
        nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
        print(f"  {c}: {nuniq}")
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
df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)
#df.to_csv(r"C:\Users\INVESTIGACION 47\Downloads\datawos_scopuseliminar2.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
