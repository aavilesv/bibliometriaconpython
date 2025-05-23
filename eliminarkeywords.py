import pandas as pd

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv("G:\\Mi unidad\\2025\\Master Almeida Monge Elka Jennifer\\data\\datawos_scopuslematizar.csv")
   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [

    "article", "human", "human experiment", "female", "male", "adult",
    "model", "performance", "young adult", "artificial intelligence",
    "control study", "major clinical study", "physiology", "product design",
    "priority journal", "brainstorm", "communication", "decision making",
    "management", "write", "child", "experience", "gender difference",
    "language", "sustainability", "task performance", "torrance test",
    "work memory", "assessment", "brain", "chatgpt", "entrepreneurship education",
    "teacher", "technology", "art", "culture", "engineering", "engineering design",
    "exercise", "faculty", "abdominal obesity", "abduction", "Alzheimer’s disease",
    "anorexia nervosa", "anemia", "antigen", "arterial hypertension", "arthritis",
    "asthma", "bipolar disorder", "blood vessel", "brain tumor", "cancer",
    "cell proliferation", "chemotherapy", "chronic pain", "cardiovascular disease",
    "coronavirus", "coronavirus disease 2019", "coronary artery", "diabetes",
    "enzyme kinetics", "gene expression", "heart failure", "hypertension",
    "immunology", "infection", "kidney failure", "leukemia", "lipid metabolism",
    "mitochondria", "nephrology", "oncology", "osteoporosis", "Parkinson’s disease",
    "pharmacology", "quantum mechanics", "radiation therapy", "renal failure",
    "rheumatology", "sepsis", "stroke", "thrombosis", "viral load", "covid 19",
    "hong kong"

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

df.to_csv("G:\\Mi unidad\\2025\\Master Almeida Monge Elka Jennifer\\data\\datawos_scopuslematizar.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
