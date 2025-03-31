import pandas as pd

# Cargar el archivo CSV

#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")
df = pd.read_csv("G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopus_unirkeywords.csv")
   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [
 "human", "article", "female", "control study", "male", "adult", "middle aged", "aged", "nonhuman", "very elderly",
    "breast cancer", "hospital", "pathology", "clinical article", "clinical feature", "nuclear magnetic resonance imaging",
    "magnetic resonance imaging", "diagnostic imaging", "diagnostic test accuracy study", "image segmentation", "covid 19",
    "coronavirus disease 2019", "clinical trial", "animal", "child", "cancer patient", "cancer diagnosis", "radiomic",
    "x ray compute tomography", "electronic health record", "clinical outcome", "brain", "human tissue", "clinical decision make",
    "clinical effectiveness", "computer assist tomography", "drug industry", "breast neoplasm", "breast tumor",
    "health care personnel", "hospital management", "human cell", "hypertension", "image enhancement", "intensive care unit",
    "isolation and purification", "least absolute shrinkage and selection operator", "patient safety", "pharmaceutic",
    "prevention and control", "case control study", "clinical decision support system", "cohort study", "contrast enhancement",
    "depression", "false negative result", "farm", "histopathology", "image classification", "image quality",
    "mild cognitive impairment", "normal human", "pathophysiology", "personalize medicine", "preoperative evaluation",
    "remote sensing", "tumor volume", "antenna", "biomarker", "camera", "cancer grading", "cattle", "cognitive defect",
    "coronary artery disease", "coronavirus", "early cancer diagnosis", "early detection of cancer", "echomammography",
    "electroencephalography", "leukocyte count", "livestock", "medical imaging", "medicine", "pharmaceutical preparation",
    "mammography", "mental health", "mining", "mobile phone", "neoplasm", "radiologist", "blood", "blood pressure",
    "blood sample", "body height", "brain region", "brain size", "cancer radiotherapy", "car driving", "clinical evaluation",
    "cytology", "diabete mellitus type 2", "disease exacerbation", "electroencephalogram", "emergency health service",
    "intrusion detection", "kinematic", "limit of detection", "lymph nod", "molecular diagnosis", "natural disaster",
    "nomogram", "non insulin dependent diabete mellitus", "outpatient department", "parkinson disease", "patient selection",
    "pharmacy service hospital", "physician", "positron emission tomography", "practice guideline",
    "real time polymerase chain reaction", "tumor", "urinalysis", "wage", "wood", "antineoplastic agent", "aquaculture",
    "atherosclerotic plaque", "cancer risk", "compute tomographic angiography", "environmental dynamism", "epidemiology",
    "erythrocyte", "evolutionary game", "farming system", "financial security", "financial system", "finland", "gas emission", "gasoline", "genotype", "geometry", "gold standard",
    "health care management", "health care planning", "health record", "health risk", "health survey", "heart disease",
    "heart rate", "hospital discharge", "hospital sector", "household", "human centric", "human robot interaction",
    "humanitarian logistic", "humanitarian supply chain", "image software", "intrusion detection system", "iron",
    "job performance", "job shop scheduling", "labor", "labor market", "language", "language processing", "lymph node",
    "lymph node metastasis", "management be", "manipulator", "medical education", "medical technology", "medication error",
    "metaverse", "micmac", "multidetector compute tomography", "multiple sclerosis", "natural language", "natural resource",
    "near infrared spectroscopy", "near neighbour", "network layer", "non small cell lung cancer", "patient care",
    "patient compliance", "patient satisfaction", "pharmaceutic", "potable water", "pregnancy", "pyrolysis", "radiology",
    "record management", "roadmap", "scientometric", "security of datum", "sensor nod", "signal noise ratio",
    "spectrum analysis", "stackelberg game", "sub saharan africa", "telehealth", "temperature", "theoretical study",
    "ultrasonography", "ultrasound", "urban area", "urinary tract infection", "vegetable", "water supply", "white matter",
    "acceptance", "account", "additive", "adipose tissue", "adverse event", "age distribution", "agricultural development",
    "agriculture 40", "alignment", "alzheimer 's disease", "ambidexterity", "animal experiment", "anti collision algorithm",
    "asthma", "australia", "automobile industry", "automobile manufacture", "axillary lymph node", "bacterial infection",
    "binary alloy", "biobank", "biomedical engineering", "bitcoin", "blood glucose", "blue economy", "brain hemorrhage",
    "cancer", "cancer chemotherapy", "capital", "carbon sequestration", "caregiver", "cement industry", "chemical analysis",
    "clinical protocol", "clinician", "colorectal cancer", "community care", "computer crime",
    "conservation of natural resource", "cryopreservation", "cryptocurrency", "daily life activity", "dairy cow", "death",
    "decentralized finance", "differential game", "digital imaging", "disease surveillance",
    "dye", "e-waste", "eco design", "effluent", "emergency care", "gene expression", "glioblastoma", "head and neck cancer",
 "hospital administration", "hyperspectral imaging", "immunology", "inpatient", "laboratory diagnosis", "lactation",
 "land use", "lung disease", "mammal", "medical history", "mental disease", "microalgae", "military operation",
 "milk production", "mineral resource", "modular construction", "neurology", "nuclear power plant", "obesity",
 "ophthalmology", "polymerase chain reaction", "prostate cancer", "public hospital", "radiological parameter",
 "robotic solution", "school child", "sleep apnea obstructive", "skin cancer", "subarachnoid hemorrhage", "urinalysis",
 "vaccine", "virology", "wastewater treatment", "symptom", "synergistic effect", "synthesis", "t2 weight image",
 "technology pharmaceutical", "test", "texture analysis", "thailand", "tissue", "track technology", "trade credit",
 "trade performance", "traditional manufacturing", "trajectory tracking", "transformational leadership", "transformer",
 "travel demand", "treatment planning", "treatment response", "tumor burden", "tumor differentiation", "tumor invasion",
 "turkey", "twitter", "ultrasonic", "unemployment", "uranium alloy", "urban planning", "urbanization", "video recording",
 "viet nam", "vietnam", "virus load", "volumetry", "vulnerability", "waste product", "waste to energy",
 "water supply system", "wavelet analysis", "wavelet transform", "wearable technology", "wi fi", "work",
 "education computing", "educational certificate", "educational programme", "eeg", "electric network analysis",
 "electric power distribution", "electric power supply", "electric power system", "electric vehicle battery",
 "electromagnetic field", "electromyography", "electronic document exchange",
 "electronic document identification system", "electronic equipment", "electronic manufacturer",
 "electronic patient record", "electrophysiology", "emergency", "emergency medicine", "emergency patient",
 "emergency service", "energy harvesting", "energy optimization", "energy use", "engineer", "engineer to order",
 "engineering management", "engineering practitioner", "engineering process",
 "environmental and social sustainability", "environmental benefit", "environmental condition",
 "environmental effect", "environmental factor", "environmental footprint", "environmental innovation",
 "environmental orientation", "environmental planning", "environmental product declaration", "environmental safety",
 "environmental strategy", "environmental turbulence", "enzyme link immunosorbent assay",
 "epidermal growth factor receptor 2", "escherichia coli", "estrus", "ethereum", "evidence base medicine",
 "evidence base practice", "extracorporeal oxygenation", "extreme learning machine", "extreme weather", "eye fundus",
 "eye protection", "failure prediction", "fall", "false positive", "family", "family firm", "family history",
 "farmer knowledge", "fast fouri transform", "fault classification", "fault tolerance", "feedstock", "femur", "fever",
 "filesystem", "filtration", "financial constraint", "financial crisis", "financial efficiency",
 "financial institution", "financial risk", "fintech company", "fishery", "flow sensor", "fluorescence",
 "fluorescence in situ hybridization", "fluorodeoxyglucose f18", "follow up study", "food intake",
 "food manufacturing", "food market", "food product", "foodborne disease", "forest", "forest 40",
 "forest operation", "forestry production", "formulate product", "fossil fuel", "fouri transform",
 "fouri transform infrared spectroscopy", "freeze thaw", "fresh food", "frontal cortex", "frontline health worker",
 "fuel cost", "full consistency method", "furniture industry", "gabapentin", "gadolinium pentetate meglumine", "gait",
 "gallium arsenide", "gamma spectrometry", "gas chromatography", "gas industry", "gastrointestinal symptom",
 "gaussian noise", "gene expression profiling", "gene frequency", "gene sequence", "gene therapy", "gestational age",
 "gesture recognition", "gi", "glasgow coma scale", "global climate", "global energy", "global population",
 "glycemic control", "china",
 "education", "energy consumption", "pharmaceutical", "pharmaceutical industry", "pharmacy", "public health",
   "telemedicine", "diagnosis", "prognosis", "mhealth", "asr", "well bad method", "construction industry",
   "digital health", "smart city", "social medium", "compute tomography", "object detection",
   "contingency theory", "customer concentration", "digital marketing", "financing constraint",
   "image processing", "human resource management", "india", "institutional pressure", "institutional theory",
   "marketing", "organizational culture", "train", "entrepreneurial orientation",
   "organizational information processing theory", "resource base view", "textile industry", "construction",
   "ergonomic", "o3-mini","systematic literature review", "literature review", "italy","systematic review","0", "o",
   "pandemic", "review",
   "thing", "health care", "major clinical study", "diagnostic accuracy", "pandemic",
   "receiver operate charact", "cohort analysis", "sensitivity and specificit", "agriculture", "o3-mini",
   "retrospective study", "sensitivity and specificity", "priority journal",
  "receiver operate characteristic", "young adult", "adolescent", "aged 80 and over",
  "disease", "healthcare"

    

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
df.to_csv("G:\\Mi unidad\\Master en administración y empresas\\articulo 2\\wos_scopuslibrería_procesadovos.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
