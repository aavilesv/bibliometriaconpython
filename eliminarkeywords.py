import pandas as pd

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv("G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\wos_scopus_reemplazado.csv")
   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [
    "alzheimer",
    "bipolar disorder",
    "bacterial foraging",
    "barium compound",
    "nanogenerator",          # Demasiado específico en física/materiales
    "triboelectric nanogenerator",
    "triboelectricity",
    "electrocardiogram",
    "electrocardiography",
    "electroencephalography",
    "aedes aegypti",          # Mosquito específico
    "betacoronaviru",         # Variante específica
    "breast neoplasm",        # Tipo de cáncer específico
    "cardiovascular disease",
    "autism spectrum disorder",
    "brain tumor",
    "mental illness",
    "acetone",                # Compuesto químico
    "carbon monoxide",
    "biochemistry",
    "animal food",
    "broiler",                # Tipo de pollo
    "animal welfare",         # Muy de nicho, a menos que el enfoque sea ese servicio público exacto
    "chemoresistive sensor",
    "gas detector",

    # --- Términos de Ingeniería o Física Excesivamente Granulares ---
    "antenna",
    "1d dilate causal cnn",   # Arquitectura de modelo extremadamente específica
    "adhesive",
    "aerodynamic",
    "casting defect",         # Proceso de manufactura muy específico
    "wavelet transform",      # Herramienta matemática muy específica
    "yolo",                   # Nombre de un modelo de visión por computadora específico
    "bert",                   # Nombre de un modelo de lenguaje específico
    "hopfield neural network",# Modelo de red neuronal antiguo y muy específico
    "turing machine",         # Concepto teórico de computación, no una aplicación
    "bin",                    # Demasiado ambiguo (contenedor, etc.)
    "gate recurrent unit",
    "adaboost",
    
    # --- Conceptos de Negocios o Sectores muy Específicos ---
    "b2b relationship",
    "b2c",
    "crm",                    # Aunque existe "CitizenRM", CRM es un término muy comercial
    "advertizing",
    "hotel",
    "tourism",                # Generalmente sector privado, no servicio público central
    "automobile manufacture",
    "marketing",
    "sale",
    
    # --- Nombres Propios, Lugares o Términos Históricos ---
    # (El estudio es sobre conceptos, no sobre casos geográficos o históricos puntuales)
    "ukraine",
    "south korea",
    "pakistan",
    "brazil",
    "estonia",                # Aunque es un caso de estudio famoso, es un nombre propio
    "birmingham england",
    "amsterdam north holland",
    "california",
    "china",
    "malaysia",
    "greece",
    "italy",
    "spain",
    "sweden",
    "scotland",
    "administration of the 12th century", # Fuera del marco temporal

    # --- Palabras Ambiguas, Genéricas o Metodológicas que no son el "qué" sino el "cómo" ---
    "article",                # Se refiere al tipo de documento, no al contenido
    "current",
    "female",                 # Característica demográfica, no un concepto del tema
    "male",
    "adult",
    "aged",
    "student",
    "job",
    "perspective",
    "future",
    "bibliometric analysis",  # Es la metodología que usas, no parte del tema en sí
    "co word analysis",       # Ídem
    "citespace",              # Ídem, es un software para el análisis
    "web of science",         # Ídem, es una base de datos
    "delphi study",           # Tipo de metodología
    "survey",
    "case study",
    "mean square error",      # Métrica técnica
    "accuracy",               # Métrica técnica
    "parameter estimation",   # Proceso técnico
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

# Guardar el DataFrame filtrado en un nuevo archivo CSV
#df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)

df.to_csv("G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\datawos_scopuseliminadas.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
