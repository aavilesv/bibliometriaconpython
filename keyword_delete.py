import pandas as pd

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv(r"G:\Mi unidad\2025\master CASTRO CASTRO ARACELLY GISELLA\data\datawos_scopuslematizar.csv")
   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [

     "forestry",
    "major clinical study",
    "retrospective study",
    "brain",
    "current",
    "model base opc",
    "on machine",
    "early warning score",
    "high school",
    "human experiment",
    "internet of thing",
    "performance base",
    "semantic",
    "offline",
    "personnel training",
    "social medium",
    "turing machine",
    "university hospital",
    "balancing",
    "bangladesh",
    "clinical deterioration",
    "clinical outcome",
    "computation theory",
    "deterioration",
    "digital transformation",
    "failure analysis",
    "heart arrest",
    "heart rate",
    "mortality",
    "motion compensation",
    "neural network computer",
    "professional aspect",
    "sepsis",
    "short term memory",
    "version control system",
    "aged",
    "middle aged",
    "young adult",
    "breathing rate",
    "code symbol",
    "diagnosis",
    "emergency ward",
    "engineering research",
    "follow up",
    "human resource management",
    "informatic",
    "lm",
    "mathematics",
    "middle school",
    "modal analysis",
    "network base",
    "observational study",
    "oxygen saturation",
    "resuscitation",
    "software testing",
    "systolic blood pressure",
    "adverse event",
    "animal",
    "artificial ventilation",
    "bangkok",
    "be education",
    "binary alloy",
    "book",
    "budapest university",
    "budget control",
    "business intelligence",
    "business intelligence in education",
    "centralise",
    "china",
    "complication",
    "component",
    "condition",
    "conventional machine",
    "critical care",
    "current situation",
    "cybersecurity",
    "data assimilation",
    "digital library",
    "disease",
    "down stream",
    "dynamic",
    "economic",
    "economic cost",
    "electronic medical record",
    "engineering degree",  # si SOLO ingeniería clínica; de lo contrario, puedes quitarla
    "emotional intelligence",
    "facebook",
    "finance",
    "financial management",
    "financial risk",
    "graph structure",     # grafos genérico sin contexto educativo
    "health care facility",
    "heart",
    "hospital",
    "hospital admission",
    "hospitalization",
    "industrial electronic",
    "industrial revolution",
    "intelligent vehicle highway system",
    "iot",
    "italy",
    "k-12",
    "k-12 education",
    "kingdom of saudi arabia",
    "knowledge management",  # genérico; si no lo usas para features, excluir
    "large amount of data",  # muy genérico
    "least square approximation",
    "length of stay",
    "machine data",
    "machinery",
    "markov decision process",
    "markov process",
    "medicine",
    "modern education",      # demasiado genérico
    "modify early warning score",  # término médico (NEWS)
    "national taiwan university",
    "network coding",
    "network layer",
    "nigerian university",
    "numerical model",
    "operation duration",
    "opinion",               # sin “mining”
    "public university",     # demasiado genérico como keyword
    "record management",
    "research study",
    "risk perception",       # fuera de foco si no es EWS
    "selection",             # genérico
    "skill",                 # genérico
    "spark",                 # herramienta genérica; quítala si no filtras por herramientas
    "statistic",             # genérico
    "stem",                  # amplio; si no filtras por dominio, excluye
    "student 's characteristic", # ruido/forma mala
    "student base",
    "student population",    # genérico demográfico
    "supplemental instruction",
    "teaching quality",      # genérico (no ML/EWS)
    "times series",          # error tipográfico duplicado de time series
    "university environment",
    "academic activity",
    "academic career",
    "academic research",
    "adaptivity",            # genérico
    "age group",
    "analysis",              # genérico
    "analysis approach",
    "analytic system",
    "application program",
    "apply informatic",
    "apply machine learning",
    "apply statistic",
    "area under curve",      # duplicada de 'area under the curve'
    "asynchronous learning", # fuera de foco si no es para predicción
    "auto encoder",          # fuera si no consideras representación
    "automate machine",      # ruido
    "bachelor 's degree",    # demográfico
    "bayes classifier",      # sin contexto (ya tienes naive/bayesian)
    "behavior detection",    # fuera si no analizas comportamiento
    "behavior pattern",      # muy genérico
    "bi directional",        # ruido
    "blending",              # ruido
    "classified",            # ruido
    "community detection",   # fuera de foco si no haces grafos
    "complex network",       # idem
    "complication",          # clínico
    "computational model",   # genérico
    "computer engineering",  # dominio específico
    "computer network",      # fuera
    "computer operating system",
    "computer system programming",
    "computing education",   # si NO segmentas por subdominio, excluye
    "condition",             # genérico
    "course",                # genérico
    "degree program",        # genérico
    "demographic group",     # ruido
    "depression",            # clínico
    "deregulation",          # política pública
    "descriptive statistic", # genérico
    "design and development",# genérico
    "diagnostic test accuracy study", # clínico
    "diastolic blood pressure",       # clínico
    "disruptive technology", # genérico
    "disruptive technology in education",
    "domain knowledge",      # genérico
    "education field",       # genérico
    "education industry",    # genérico
    "education institution", # duplicado genérico
    "education programme",   # genérico
    "educational activity",  # genérico
    "educational context",   # genérico
    "educational material",  # genérico
    "educational model",     # genérico
    "educational modelling", # genérico
    "educational performance", # genérico
    "educational problem",   # genérico
    "educational program",   # genérico
    "educational sector",    # duplicado de education sector
    "ethical technology",    # genérico
    "experimental group",    # diseño experimental genérico
    "exploratory study",     # genérico
    "face to face",          # genérico
    "fairness",              # demasiado amplio (usa 'algorithmic fairness')
    "feature space",         # genérico
    "feature vector",        # genérico
    "feedback to student",   # intervención, no predicción
    "filtration",            # ruido
    "finance",               # fuera
    "financial management",
    "financial risk",
    "fold cross validation", # duplicado de cross validation
    "formal education",      # genérico
    "game design",           # fuera
    "graduation",            # genérico (usa graduation rate)
    "graph representation",  # fuera si no haces GNN
    "health",                # fuera
    "high degree of accuracy", # ruido
    "high level language",   # fuera
    "in hospital mortality", # clínico
    "independent variable",  # genérico
    "individual difference", # genérico
    "inference engine",      # fuera
    "inform consent",        # clínico/ética biomédica
    "information",           # genérico
    "information analysis",  # genérico
    "information technology sector",
    "informatization",
    "institutional framework",
    "instructional material",
    "intelligent prediction",# ruido
    "interaction behavior",  # genérico
    "intermethod comparison",# clínico/estadístico genérico
    "internet access",       # variable contextual; suele meter ruido
    "italy",                 # país
    "item analysis",         # evaluación educativa general (si no la usas)
    "kdd",                   # congreso; si no filtras por venues, excluye
    "knowledge",             # genérico
    "knowledge graph",       # fuera si no usas KG
    "large scale",           # genérico
    "learning ability",      # genérico
    "learning capability",   # genérico
    "learning difficulty",   # genérico
    "learning efficiency",   # genérico
    "learning framework",    # genérico
    "learning resource",     # genérico
    "learning technology",   # genérico
    "learning tool",         # genérico
    "life long learning",    # fuera de foco
    "mean",                  # ruido
    "modeling language",     # fuera
    "multi layer perceptron",# duplicada de multilayer perceptron
    "multi modal",           # genérico (usa multi modal learning/data)
    "new approach",          # ruido
    "numerical method",      # fuera
    "online environment",    # genérico
    "online high education", # forma rara
    "open system",           # fuera
    "operation duration",    # fuera
    "opinion",               # ya agregado arriba
    "pathophysiology",       # clínico
    "patient monitoring",    # clínico
    "performance factor",    # genérico
    "performance metrice",   # error ortográfico
    "personal information",  # privacidad (demasiado amplio)
    "prediction base",       # ruido
    "prior knowledge",       # genérico
    "probability",           # genérico
    "problem base learning", # metodología; fuera si no segmentas
    "procedure",             # genérico
    "quality assurance",     # gestión; fuera
    "receiver operating characteristic curve", # duplicado de roc curve
    "reproducibility",       # si no anotas como variable binaria, puede meter ruido
    "research study",        # genérico
    "root mean square error",# duplicado de mean square error
    "selection",             # genérico (ya arriba)
    "software testing",      # ya arriba
    "statistical method",    # genérico
    "statistical model",     # genérico
    "student data",          # genérico
    "student education",     # genérico
    "student enrollment",    # genérico
    "student feedback",      # intervención, no predicción
    "technology enhance learning", # forma rara (TEL genérico)
    "university sector",     # genérico
    "website",               # fuera
    "accreditation",         # gestión
    "attention",             # genérico (mantén “attention mechanism”)
    "augmented reality",     # fuera del foco
    "budget control",        # ya arriba
    "causal inference",      # fuera si no lo usas
    "classification accuracy",# redundante (usa accuracy)
    "classroom learning",     # genérico
    "community detection",    # ya arriba
    "computational intelligence", # genérico
    "computer assist instruction", # duplicado/variante
    "computer vision",        # fuera si no usas CV
    "computing education",    # ya marcado arriba como excluible
    "correlation",            # genérico
    "correlation coefficient",# genérico
    "counterfactual",         # fuera si no usas XAI causal
    "course",                 # ya arriba
    "course design",          # fuera
    "covid-19 challenge",     # si no filtras por pandemia, excluye
    "cross validation method",# duplicado de cross validation
    "cumulative grade point average", # dup de grade point average / cgpa
    "curricular complexity",  # fuera si no la analizas
    "data classification",    # genérico
    "data collection",        # genérico
    "data collection process",# genérico
    "data consistency",       # genérico
    "data reduction",         # genérico
    "data reliability",       # genérico
    "data representation",    # genérico
    "data visualisation",     # duplicado de data visualization
    "dataset",                # genérico
    "datum transformation",   # ruido
    "decision making process",# genérico
    "decision support",       # genérico
    "deep learning dl",       # duplicado de deep learning
    "demographic data",       # genérico
    "demographic factor",     # genérico
    "digital footprint",      # fuera
    "digital library",        # ya arriba
    "dispositional learning analytic", # inclúyelo si mapeas dispositional LA; si no, excluye
    "disruptive technology in education", # ya arriba
    "education field",        # ya arriba
    "educational analytic",   # forma ruidosa
    "educational data",       # genérico
    "educational evaluation", # genérico
    "electronic health record",# clínico
    "emotional intelligence", # ya arriba
    "engineering degree",     # ya arriba
    "evaluation criterion for data prediction", # ruidoso
    "experimental evaluation",# genérico
    "extreme gradient boosting", # si ya tienes xgboost/catboost, puedes excluir
    "feature importance",     # genérico (puede quedarse, pero tiende a ruido)
    "felder silverman learning style model fslsm", # fuera del foco
    "financial management",   # ya arriba
    "formal education",       # ya arriba
    "game design",            # ya arriba
    "gradient method",        # genérico
    "graph convolutional network", # mantén si usas GCN; si no, excluye
    "hide markov model",      # si no usas HMM explícitamente, excluye
    "high accuracy",          # ruido
    "important feature",      # ruido
    "improve decision tree",  # ruido
    "improve smote",          # ruido
    "independent variable",   # ya arriba
    "information gain",       # puede meter ruido
    "information technology sector", # ya arriba
    "institution of high education",  # forma ruidosa
    "instructional material", # ya arriba
    "interactive learning",   # genérico
    "intermethod comparison", # ya arriba
    "item analysis",          # ya arriba
    "knowledge",              # ya arriba
    "learning and teaching",  # genérico
    "learning classifier",    # ruidoso
    "learning course",        # ruidoso
    "learning design",        # fuera
    "learning method",        # duplicado de learning technique
    "learning path",          # fuera si no mapeas rutas
    "learning situation",     # ruidoso
    "linear model",           # genérico
    "mapping",                # genérico
    "mathematical model",     # genérico
    "modern education",       # ya arriba
    "multi dimensional analysis", # genérico
    "multilayer",             # ruidoso
    "multiple linear regression", # si ya mantienes linear/logistic/regression analysis
    "nearest neighbours nn",  # ruido (dup de knn/nearest neighbor)
    "network layer",          # ya arriba
    "nn",                     # ruido (abreviatura ambigua)
    "numerical model",        # ya arriba
    "online class",           # genérico
    "online platform",        # genérico
    "online teaching",        # genérico
    "ontology 's",            # error/ruido
    "open system",            # ya arriba
    "operation duration",     # ya arriba
    "opinion",                # ya arriba
    "open university learning analytic dataset", # duplicado de oulad / oulad dataset
    "article"

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

df['Index Keywords'] = df['Index Keywords'].apply(filter_unique)
df['Author Keywords'] = df['Author Keywords'].apply(filter_unique)
# Guardar el DataFrame filtrado en un nuevo archivo CSV
#df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)

df.to_csv(r"G:\Mi unidad\2025\master CASTRO CASTRO ARACELLY GISELLA\data\datawos_scopusdelete.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
