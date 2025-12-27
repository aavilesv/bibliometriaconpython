import pandas as pd
import re

# Cargar el archivo CSV
df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

#df = pd.read_csv(r"C:\Users\INVESTIGACION 47\Downloads\scopusdata.csv")
KW_COLS     = ["Author Keywords", "Index Keywords"]
# Diccionario de palabras clave a reemplazar: clave = palabra a buscar (en minúsculas), valor = palabra de reemplazo
palabras_clave_reemplazo = {
 # ===== ARTIFICIAL INTELLIGENCE =====
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "artificial-intelligence": "artificial intelligence",
    "artificial intelligence (ai)": "artificial intelligence",
    "artificial intelligence technologies": "artificial intelligence",
    "artificial intelligence tools": "artificial intelligence",
    "artificial intelligence technology": "artificial intelligence",

    # ===== MACHINE LEARNING =====
    "machine learning": "machine learning",
    "machine-learning": "machine learning",
    "machine learning (ml)": "machine learning",
    "machine learning algorithms": "machine learning",
    "supervised machine learning": "machine learning",

    # ===== DEEP LEARNING / NEURAL NETWORKS =====
    "deep learning": "deep learning",
    "deep-learning": "deep learning",
    "neural network": "neural networks",
    "neural networks": "neural networks",
    "artificial neural network": "neural networks",
    "artificial neural networks": "neural networks",
    "convolutional neural network": "neural networks",
    "convolutional neural networks": "neural networks",
    "deep neural network": "neural networks",
    "long short-term memory": "neural networks",

    # ===== GENERATIVE AI / LLMs / CHATGPT =====
    "generative artificial intelligence": "generative ai",
    "generative artificial intelligence (genai)": "generative ai",
    "generative ai": "generative ai",
    "genai": "generative ai",

    "large language model": "large language models",
    "large language models": "large language models",
    "llm": "large language models",
    "llms": "large language models",

    "chatgpt": "chatgpt",
    "chat gpt": "chatgpt",
    "chat-gpt": "chatgpt",
    "chatbots": "chatbot",
    "chatbot": "chatbot",
    "conversational agents": "chatbot",

    # ===== AI IN EDUCATION =====
    "ai in education": "artificial intelligence in education",
    "aied": "artificial intelligence in education",
    "artificial intelligence in education": "artificial intelligence in education",

    "ai in higher education": "artificial intelligence in higher education",
    "artificial intelligence in higher education": "artificial intelligence in higher education",

    # ===== EDUCATION CONTEXT =====
    "higher education": "higher education",
    "higher-education": "higher education",
    "high educations": "higher education",
    "tertiary education": "higher education",
    "postsecondary education": "higher education",

    "university": "universities",
    "universities": "universities",
    "college": "universities",
    "colleges and universities": "universities",
    "higher education institutions": "higher education institutions",
    "higher education institution": "higher education institutions",
    "higher education institutions (heis)": "higher education institutions",

    "student": "students",
    "students": "students",
    "university students": "students",
    "higher education students": "students",
    "college students": "students",
    "undergraduate students": "students",
    "university-students": "students",

    "teacher": "teachers",
    "teachers": "teachers",
    "faculty": "teachers",
    "teacher education": "teacher education",

    # ===== LEARNING MODES =====
    "e-learning": "e-learning",
    "e - learning": "e-learning",
    "electronic learning": "e-learning",
    "digital learning": "e-learning",

    "online learning": "online learning",
    "online education": "online learning",
    "distance learning": "distance education",
    "distance education": "distance education",

    "blended learning": "blended learning",
    "flipped classroom": "flipped classroom",
    "gamification": "gamification",

    # ===== ANALYTICS / DATA =====
    "learning analytics": "learning analytics",
    "learning analytic": "learning analytics",
    "educational data mining": "educational data mining",
    "education data mining": "educational data mining",
    "data mining": "data mining",
    "predictive analytics": "predictive analytics",
    "sentiment analysis": "sentiment analysis",
    "text mining": "text mining",
    "natural language processing": "natural language processing",

    # ===== PERFORMANCE / OUTCOMES =====
    "academic performance": "academic performance",
    "student performance": "academic performance",
    "student performance prediction": "academic performance",
    "academic achievement": "academic achievement",
    "academic success": "academic achievement",
    "student success": "academic achievement",
    "student retention": "student retention",
    "student dropout": "student dropout",
    "dropout prediction": "student dropout",

    # ===== ADOPTION MODELS =====
    "technology acceptance": "technology acceptance",
    "technology acceptance model": "tam",
    "tam": "tam",
    "utaut": "utaut",
    "utaut2": "utaut",
    "behavioral intention": "behavioral intention",
    "perceived usefulness": "perceived usefulness",
    "perceived ease of use": "perceived ease of use",

    # ===== ETHICS =====
    "ai ethics": "ai ethics",
    "academic integrity": "academic integrity",
    "plagiarism": "plagiarism",
    "privacy": "privacy",
        # ===== IA / ML / MODELOS =====
    "ai technologies": "artificial intelligence",
    "artificial intelligence in higher education": "artificial intelligence in higher education",
    "artificial intelligence education": "artificial intelligence in education",
    "artificial intelligence in education (aied)": "artificial intelligence in education",
    "artificial intelligence literacy": "ai literacy",

    "machine learning models": "machine learning",
    "machine learning techniques": "machine learning",
    "transfer learning": "machine learning",
    "ensemble learning": "machine learning",
    "supervised machine learning": "machine learning",

    "deep learning approach": "deep learning",
    "deep-learning": "deep learning",
    "deep neural networks": "neural networks",
    "multilayer perceptron": "neural networks",
    "lstm": "neural networks",
    "long short-term memory (lstm)": "neural networks",

    "support vectors machine": "support vector machines",
    "svm": "support vector machines",
    "xgboost": "ensemble models",
    "genetic algorithm": "optimization algorithms",
    "genetic algorithms": "optimization algorithms",
    "k-means": "clustering algorithms",
    "knn": "classification algorithms",
    "naïve bayes": "classification algorithms",
    "random forest": "ensemble models",
    "boosting": "ensemble models",
    "stacking": "ensemble models",

    # ===== GENAI / LLM =====
    "gpt-3": "large language models",
    "generative pre-trained transformer": "large language models",
    "generative ai (genai)": "generative ai",
    "language model": "large language models",
    "large language model": "large language models",

    "ai chatbot": "chatbot",
    "ai chatbots": "chatbot",
    "virtual assistant": "chatbot",

    # ===== EDUCATION CONTEXT =====
    "higher education (he)": "higher education",
    "post-secondary education": "higher education",
    "undergraduate education": "higher education",
    "online higher education": "higher education",

    "university student": "students",
    "college-students": "students",
    "students perceptions": "student perceptions",
    "student perception": "student perceptions",
    "students’ perceptions": "student perceptions",

    "teachers": "teachers",
    "university teachers": "teachers",
    "faculty members": "teachers",
    "preservice teachers": "teachers",

    # ===== LEARNING MODES =====
    "education, distance": "distance education",
    "virtual learning environment": "virtual learning environments",
    "virtual learning environments": "virtual learning environments",
    "online learning environment": "online learning",
    "technology-enhanced learning": "technology enhanced learning",
    "technology enhanced learning": "technology enhanced learning",
    "hybrid learning": "blended learning",
    "smart learning": "smart learning",

    # ===== ANALYTICS / DATA =====
    "network analysis": "network analysis",
    "topic modeling": "topic modeling",
    "topic modelling": "topic modeling",
    "text classification": "text mining",
    "nlp": "natural language processing",
    "natural language processing (nlp)": "natural language processing",
    "opinion mining": "sentiment analysis",
    "data visualization": "data visualization",

    # ===== PERFORMANCE / OUTCOMES =====
    "students performance": "academic performance",
    "student academic performance": "academic performance",
    "academic achievement": "academic achievement",
    "academic performance prediction": "academic performance",
    "early prediction": "early warning systems",
    "prediction model": "predictive models",
    "prediction modelling": "predictive models",
    "predictive modeling": "predictive models",

    "student dropout": "student dropout",
    "university dropout": "student dropout",
    "school dropout": "student dropout",

    # ===== ADOPTION / BEHAVIOR =====
    "perceived ease": "perceived ease of use",
    "technology acceptance model (tam)": "tam",
    "utaut model": "utaut",
    "task-technology fit": "task-technology fit",
    "intention to use": "behavioral intention",

    # ===== PEDAGOGY =====
    "active learning strategies": "active learning",
    "active methodologies": "active learning",
    "project based learning": "project-based learning",
    "problem based learning": "problem-based learning",
    "team-based learning": "collaborative learning",
    "community of inquiry": "community of inquiry",

    # ===== SYSTEMS / PLATFORMS =====
    "learning management system (lms)": "learning management systems",
    "moodle": "learning management systems",
    "virtual learning": "virtual learning environments",

    # ===== ETHICS =====
    "academic dishonesty": "academic integrity",
    "academic misconduct": "academic integrity",
    "ethical implications": "ai ethics",
    "ethical considerations": "ai ethics",
}







def reemplazar_palabras_clave(column, diccionario_reemplazo):
    """
    Recorre cada celda de la columna, separa los términos (suponiendo que estén separados por ';'),
    y reemplaza aquellos que coincidan (ignorando mayúsculas/minúsculas) por el valor correspondiente del diccionario.
    """
    def process_cell(cell):
        # Si la celda es una cadena, separamos usando el delimitador ';'
        if isinstance(cell, str):
            terminos = [termino.strip() for termino in cell.split(';') if termino.strip()]
        # Si ya es una lista, la usamos directamente
        elif isinstance(cell, list):
            terminos = [str(termino).strip() for termino in cell if str(termino).strip()]
        else:
            terminos = []
        
        terminos_modificados = []
        for termino in terminos:
            # Convertimos el término a minúsculas para la comparación
            termino_lower = termino.lower()
            if termino_lower in diccionario_reemplazo:
                # Reemplazamos por el valor definido en el diccionario
                terminos_modificados.append(diccionario_reemplazo[termino_lower])
            else:
                terminos_modificados.append(termino)
        return '; '.join(terminos_modificados)
    
    return column.apply(process_cell)
print("Antes (recuento únicos):")
for c in KW_COLS:
    if c in df.columns:
        nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
        print(f"  {c}: {nuniq}")

# Aplicar la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = reemplazar_palabras_clave(df['Index Keywords'], palabras_clave_reemplazo)
df['Author Keywords'] = reemplazar_palabras_clave(df['Author Keywords'], palabras_clave_reemplazo)
#df['bothKeywords'] =  reemplazar_palabras_clave(df['bothKeywords'], palabras_clave_reemplazo)
# --- A PARTIR DE AQUÍ, EL CÓDIGO NUEVO PARA REEMPLAZOS PARCIALES ---
print("\nDespués (recuento únicos):")
for c in KW_COLS:
    if c in df.columns:
        nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
        print(f"  {c}: {nuniq}")
def reemplazar_parciales(column, patrones):
    """
    Recorre cada celda de la columna, y por cada patrón (regex) en 'patrones',
    realiza la sustitución indicada.
    - 'patrones' debe ser una lista de tuplas (pattern, replacement).
    - Se ignoran mayúsculas/minúsculas (flags=re.IGNORECASE).
    """
    def process_cell(cell):
        if isinstance(cell, str):
            # Aplica todos los patrones de reemplazo parcial
            for patron, nuevo_texto in patrones:
                cell = re.sub(patron, nuevo_texto, cell, flags=re.IGNORECASE)
        return cell

    return column.apply(process_cell)

# Ejemplo de un arreglo de reemplazos parciales
# Cada tupla es (expresión_regular, texto_reemplazo)
# Aquí solo se incluye 'datum' -> 'data', pero puedes añadir más.
patrones_parciales = [
    (r'datum', 'data'),  # Reemplaza 'datum' donde aparezca (ignora mayúsculas)
    # Si necesitas más reemplazos parciales:
    # (r'algunaSubcadena', 'otroTexto'),
    # (r'pattern', 'replacement'),
    # ...
]
# 2) Después, los reemplazos parciales:
df['bothKeywords'] = reemplazar_parciales(df['bothKeywords'], patrones_parciales)
#df['Index Keywords'] = reemplazar_parciales(df['Index Keywords'], patrones_parciales)
#df['Author Keywords'] = reemplazar_parciales(df['Author Keywords'], patrones_parciales)
# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)

#df.to_csv(r"C:\Users\INVESTIGACION 47\Downloads\datawos_scopusreplace.csv", index=False)
print("Palabras clave reemplazadas y nuevo archivo guardado.")