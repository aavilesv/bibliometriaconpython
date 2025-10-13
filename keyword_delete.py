import pandas as pd

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replace.csv")
   # Lista de palabras clave a eliminar (en minúsculas)
palabras_clave_a_eliminar = [

    "article","paper","publication","publications","review","literature review","systematic review",
    "study","studies","case study","case-studies","case report","research","research papers",
    "method","methods","methodology","methodologies","approach","approaches","model","models",
    "framework","frameworks","protocol","protocols","process","processes","procedure","procedures",
    "design","designs","analysis","analyses","statistical analysis","descriptive analysis",
    "evaluation","evaluations","assessment","assessments","measurement","measurements","metrics",
    "results","findings","discussion","conclusions","implications","introduction",
    # instrumentos/diseños de estudio
    "questionnaire","questionnaires","survey","surveys","focus groups","interview","interviews",
    "open-ended questions","qualitative research","quantitative research","mixed methods",
    "cross-sectional study","cross-sectional studies","randomized controlled trial",
    "randomized controlled trials as topic","pretest posttest design","pilot study",
    # etiquetas genéricas de rendimiento
    "performance","impact","impacts","outcomes","outcome study","effectiveness","efficiency",
    # términos administrativos
    "management","organization","organizations","policy","policies","strategy","strategies","modeling",
    # ruido de formato/colecciones
    "web of science","scopus","cinahl","psycinfo","open access","openaccess","keywords",
    # conectores / comodines temáticos
    "technology","technologies","information","systems","services","environment","context",
    "implementation","application","applications","factors","challenges","barriers","drivers",
    "trends","issues","aspects","dimensions","features","characteristics","perspectives","theory",
    "theories","conceptual framework","conceptual frameworks",
       "united states","uk","canada","australia","spain","china","india","indonesia","saudi arabia",
        "vietnam","malaysia","pakistan","jordan","portugal","france","germany","italy","mexico",
        "latin america","europe","western europe","middle east","south-east asia","southeast asia",
        "africa","sub-saharan africa","nordic countries","ireland","turkey","uae","ukraine",
        "west bengal","kerala","karnataka","tamil nadu","jammu and kashmir","henan province",
        "yangtze river economic belt","western china",
            "cytotoxicity", "cytotoxicity  immunologic", "dipeptidyl carboxypeptidase inhibitor",
    "dipeptidyl peptidase iv inhibitor", "embryonic stages", "endotracheal intubation",
    "diabetes care", "diabetes education", "neoplasm", "neoplasms", "pneumonia  viral",
    "drug efficacy", "drug misuse", "drug utilization", "resuscitation",
        "frailty", "frailty prevention",
    "functional near-infrared spectroscopy",
    "hemoglobin a1c",
    "histocompatibility antigens class i", "hla antigen", "hla antigens", "hla-e antigen",
    "immunoglobulins", "immunomodulation",
    "insulin",
    "infectious diseases", "infectious disease medicine",
    "inpatients", "intensive care unit", "intensive care units",
    "hospital admission", "hospital emergency service", "hospital pharmacy",
    "gynecologist",
    "hydroxychloroquine", "hydrocortisone",
    "heart surgery", "heart beats",
    "endotracheal intubation",
    "hydrocarbons", "gasoline",  # fuera de foco educativo/liderazgo
    "diabetes care", "diabetes education",  # si tu foco no es salud,
        # Tokens demasiado generales o de “relleno”
    "key","form","forms","forum","forums","future","looking","leave",
    "major factors","main tasks","mapping","mapping method","methods","models","mechanisms",
    "processes","programs","research","results","losses",
    "level management","levels of analysis",
    "learning and teachings","learning and teaching methodologies",
    "media","news",
    "online environment","online channels","online products",
    "market","business","companies","company",
    "people","workers","students","lecturer",
    "country","region","world",
    "quality","efficiency",
        # Demasiado generales/ambivalentes para co-ocurrencias útiles
    "opportunity","operations","operations services","operation management",
    "optimal solutions","optimal systems","optimistic","place","policies","policy",
    "practice","practices","programs","projects","research design","research focus",
    "recommendations","regional development","regions of russia","population","presence",
    "ranking","requirements","representation","resources","review comments","risk factor",
    "roadmap","roads","salary","sampling","schedule flexibility","science","season",
    "search","second phase","selection",
    # Ruido clínico/básico que se aleja del dominio edu/gestión/liderazgo digital
    "ophthalmology","oral surgery","orthopedics","peptide","peptides",
    "physiology","protein","proteins","pathogenesis","plasmodesma",
    # Siglas/abrevs ambiguas de 2-3 letras
    "pm","plcs","plma","psm systems",
    



      # Demasiado genéricos/ruido
    "system","systems","structure","structures","statistics","success","strategies",
    "study design","theoretical research","things","tool","tools","temporary","tenure",
    "transition","translation","understanding","usage situations","users","value streams",
    # Lugares/propios (no términos conceptuales)
    "shandong","singapore","slovenia","southwest china","spanish iberia","uae","uk",
    "texas","uganda","turkey","trinidad and tobago","tanzania","tripura","siberia",
    "st  mary's college of meycauayan","st mary college of meycauayan",
    "the university of southern mississippi","university of edinburgh",
    "university of nebraska-lincoln","us military","united states air force",
    # Nombres propios/marcas/entidades
    "sitecore","petronas","telenor","twitter network","theodore roosevelt","theresa may",
    "shuyan wang","smcm integrated student activities (sis)","smcm integrated student activity",
    # Biomédico/clinico fuera de dominio principal
    "simvastatin","sodium glucose cotransporter 2 inhibitor","vascular surgery",
    "thorax radiography","thorax surgery","tumor board","psychiatry","psychiatric nurses",
    "pediatric surgery","vascular access","therapeutic target","therapeutic research",
    # Ambiguas/medios/ruidosas
    "sports","sport","video games","games","speedspeed","things","season",
    "standard versions","standard organization",
      # Lugares / regiones / topónimos
    "washington","west bengal","west siberia","western balkan leadership",
    "western china","western europe","western hemisphere","western pacific region",
    "yangtze river economic belt",

    # Industrias/materiales muy específicos (ruido de dominio)
    "zinc metallurgy","zinc mine","zinc mines","warehouses","ward",

    # Nombres propios/marcas/eventos/palabras sueltas
    "whatsapp","whatsapp messenger","zoom","wasta","wenurses","warp-pls"  # <- mantener solo en preserve
    # Nota: aunque existan como plataformas populares, suelen sesgar co-ocurrencias temáticas
    ,

    # Demasiado genéricas/ruidosas
    "x","weight","weight bias","wages","water flow","water movements","water supply",
    "world-class researches","world class university","work in progress","work form",
    "working","working systems","working life","working professionals","workers",
    "worker","workflow","work sites","work form","forms","form",

    # Biomédico/virus (fuera de foco de liderazgo virtual)
    "virus leadership theory","virus spreading","virus theory","women's health",

    # Militar/casos muy específicos
    "warfighters",

    # Taxonomías dudosas/ruido
    "wroclow taxonomy",
     
        

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

df.to_csv(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replace.csv", index=False)
print("Palabras clave específicas eliminadas y nuevo archivo guardado.")
