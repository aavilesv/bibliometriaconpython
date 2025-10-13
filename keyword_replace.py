import pandas as pd
import re

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1.csv")
KW_COLS     = ["Author Keywords", "Index Keywords"]
# Diccionario de palabras clave a reemplazar: clave = palabra a buscar (en minúsculas), valor = palabra de reemplazo
palabras_clave_reemplazo = {
    "organisational": "organizational",
    "universities": "university",
    "students": "student",
    "teachers": "teacher",
    "employees": "employee",
    "digitisation": "digitization",
      "well being": "well-being",
    "cyber security": "cybersecurity",
    "construction 4 0": "construction 4.0",
    "industry 5 0": "industry 5.0",
    "on-line": "online",
    "on-line communities": "online communities",
    "on-line communication": "online communication",
    "on-line education": "online education",
    "organisational": "organizational",
    "organisational change": "organizational change",
    "organisational sustainability": "organizational sustainability",
    "digital competences": "digital competence",
    "digital citizenships": "digital citizenship",
    "small-and-medium enterprise": "small and medium-sized enterprise",
    "small - and medium - sized enterprises": "small and medium-sized enterprise",
    "small and medium-sized enterprises": "small and medium-sized enterprise",
    "smartpls": "SmartPLS",
    "pls": "PLS",
    "pls-sem": "PLS-SEM",
    "leader–member exchange (lmx)": "leader–member exchange (lmx)",
    "social network services": "social networking sites",   # si quieres consolidar
    "digital leadership competencies": "digital leadership competence",
    "digital leadership skills": "digital leadership skill",
    "leadership practices": "leadership practice",
    "team leaders": "team leader",
    "team members": "team member",
    "universities": "university",
    "students": "student",
    "teachers": "teacher",
       "work–life balance": "work-life balance",
    "“virtuous burden”": "virtuous burden",
    "eleadership": "e-leadership",
    "-leadership": "e-leadership",
    "4 0 era": "4.0 era",
    "construction 4 0": "construction 4.0",
    "industry 5 0": "industry 5.0",
    "cyber security": "cybersecurity",
    "on-line": "online",
    "on-line communication": "online communication",
    "on-line collaborations": "online collaborations",
    "on-line education": "online education",
    "organisational": "organizational",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "digitisation": "digitization",
    "small-and-medium enterprise": "small and medium-sized enterprise",
    "small and medium-sized enterprises": "small and medium-sized enterprise",
    "vuca": "vuca environment",
    "sem": "structural equation modeling",
    "3d/4d/5d printing": "3d printing",
    "1:1 coverage of digital device": "one-to-one device coverage",
    "1:1 coverage of digital devices": "one-to-one device coverage",
    "covid- 19 pandemic": "covid-19 pandemic",
    "covid19": "covid-19",
      # guiones / tipografía
    "cultural-diversity": "cultural diversity",
    "cyber physical systems": "cyber-physical systems",
    "digital-leadership": "digital leadership",
    "digital-leadership capability": "digital leadership capability",
    "e-leadership": "e-leadership",
    "e–leadership": "e-leadership",
    "e-leadersip": "e-leadership",
    "work–life balance": "work-life balance",
    "customer complaining behaviour": "customer complaining behavior",
    "education 4 0": "education 4.0",
    "era 4 0": "era 4.0",
    "on-line": "online",
    "distance-learning": "distance learning",
    "distance-learning education": "distance learning education",

    # variantes ortográficas
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "digitisation": "digitization",

    # data-driven
    "data driven": "data-driven",
    "data based transparency": "data-based transparency",

    # plurales a singular (si adoptas singular canónico)
    "cross-cultural organizations": "cross-cultural organization",
    "customers' satisfaction": "customer satisfaction",
    "digital badges": "digital badge",
    "digital museums": "digital museum",
    "digital twins": "digital twin",

    # covid spacing
    "covid- 19 pandemic": "covid-19 pandemic",
       # Ortografía/variante británica → US
    "feedback-seeking behaviour": "feedback seeking behavior",
    "job-satisfaction": "job satisfaction",
    "job crafting behaviours": "job crafting behaviors",
    "gender-differences": "gender differences",
    "generational-differences": "generational differences",
    "health‐oriented leadership": "health-oriented leadership",  # guion normal

    # Guiones y compuestos
    "higher-education": "higher education",
    "information-systems": "information systems",
    "information-technology": "information technology",
    "geographically-distributed teams": "geographically distributed teams",
    "intra–team communication": "intra-team communication",
    "inter–team communication": "inter-team communication",

    # Industry 4.0 y variantes
    "industry 4 0": "industry 4.0",
    "ir4 0": "industry 4.0",
    "fourth industrial revolution": "fourth industrial revolution",

    # Variantes con espacios extra / signos raros
    "covid- 19 pandemic": "covid-19 pandemic",

    # Plurales a canónico (si usas singular en keywords)
    "higher-education leadership": "higher education leadership",
    "health information systems (his)": "health information systems",
    "electronic health record (ehrs)": "electronic health records (ehrs)",

    # Mayúsculas y consistencia
    "fsqca3 0software": "fsqca 3.0 software",
      # k-means y variaciones
    "k-mean algorithms": "k-means clustering",
    "k-means cluster": "k-means clustering",
    "k-means clusters": "k-means clustering",
    # LDA y variantes
    "latent dirichlet allocation analyze": "Latent Dirichlet Allocation",
    "latent dirichlet allocation": "Latent Dirichlet Allocation",
    # MOOCs
    "massive open online course": "MOOCs",
    "massive open online courses": "MOOCs",
    "mooc": "MOOCs",
    "moocs": "MOOCs",
    # LMX familia
    "leader member exchange": "leader–member exchange (LMX)",
    "leader-member exchange": "leader–member exchange (LMX)",
    "leader-member exchange theories": "leader–member exchange (LMX)",
    "lmx theory": "leader–member exchange (LMX)",
    "lmx configurations": "leader–member exchange (LMX)",
    "leader-member exchange differentiation": "leader–member exchange (LMX)",
    "leader-member exchange relationship": "leader–member exchange (LMX)",
    "leader-member exchange social comparison (lmxsc)": "LMXSC",
    "leader–member exchange (lmx)": "leader–member exchange (LMX)",
    # Typos/estilo
    "mass-media": "mass media",
    "media usage": "media use",
    "keyword—digital leaderhip": "digital leadership",
    "keyword—digital culture": "digital culture",
    "keywords—digital culture": "digital culture",
    "keyword-clustering": "keyword clustering",
    "higher-education": "higher education",
    "hong-kong": "Hong Kong",
    "online programmes": "online programs",
    "leadership/authority": "leadership authority",
    "leader-ship": "leadership",
    "knowledge oriented leadership": "knowledge-oriented leadership",
    "knowledge based development": "knowledge-based development",
    "knowledge based interaction": "knowledge-based interaction",
    # Abreviaciones útiles
    "learning management systems": "LMS",
    # Online leadership/opinion
    "online opinion leader": "online opinion leadership",
    "online leaders": "online leadership",
    # m-learning / m-leaders (decisión práctica)
    "m-learning": "mobile learning",
    "m-leaders": "mobile leadership",
         # Variantes ortográficas / UK→US / plurales coherentes
    "online training programme": "online training program",
    "openaccess": "Open Access",
    "open educational resources (oer)": "Open Educational Resources (OER)",
    "open distance digital education (odde)": "Open Distance Digital Education (ODDE)",
    "open university ecosystem": "Open University",
    "open universit": "Open University",
    "open source project": "open-source project",
    "open source": "open source",
    "open sources": "open source",
    "open-plan offices": "open plan offices",
    "organisational behaviour": "organizational behavior",
    "organisational citizenship behaviours": "organizational citizenship behavior (OCB)",
    "organizational citizenship behaviour": "organizational citizenship behavior (OCB)",
    "organizational citizenship behaviour (ocb)": "organizational citizenship behavior (OCB)",
    "organisational commitment": "organizational commitment",
    "organisational culture and values": "organizational culture and values",
    "organisational environment": "organizational environment",
    "organisational innovativeness": "organizational innovativeness",
    "organisational knowledge capabilities": "organizational knowledge capabilities",
    "organisational learning culture": "organizational learning culture",
    "organisational objectives": "organizational objectives",
    "organisational performance": "organizational performance",
    "organisational politics": "organizational politics",
    "organisational structure": "organizational structure",
    "organisation stability": "organizational stability",
    "organization 5 0": "Organization 5.0",
    "organization’s excellence": "organizational excellence",
    "organizational-change": "organizational change",
    "organizational resiliency": "organizational resilience",
    "open/virtual laboratories": "online/virtual laboratories",

    # Palabras compuestas/hífen y consistencia
    "performance-base leadership": "performance-based leadership",
    "performance based": "performance-based",
    "problem based learning": "Problem-Based Learning",
    "project based learning": "Project-Based Learning",

    # eWOM
    "online word-of-mouth": "electronic word-of-mouth (eWOM)",
    "positive word of mouth": "positive eWOM",

    # PLS / variantes
    "partial least square (pls)": "Partial Least Squares (PLS)",
    "partial least squares regression": "PLS regression",
    "pls-ann": "PLS-ANN",
    "pls-artificial neural network": "PLS-ANN",

    # PSO / variantes
    "particle swarm optimization": "Particle Swarm Optimization (PSO)",
    "particle swarm optimization (pso)": "Particle Swarm Optimization (PSO)",

    # RFID / variantes
    "rfidradio frequency identification": "RFID (radio frequency identification)",

    # Fits y trims
    "person – job fit": "person–job fit",
    "person–organization (p–o) fit": "person–organization (P–O) fit",

    # Post-COVID consistencia
    "post covid-19": "post-COVID-19",
    "post covid-19 era": "post-COVID-19 era",
    "post-pandemic workplace": "postpandemic workplace",

    # Otros comunes
    "operation management": "operations management",
    "operations services": "operational services",
    "organizational digital mastery": "organizational digital maturity",
    "privacy by design": "Privacy by Design",
    "platform digitization capability": "platform digitalization capability",
    "platform strategies": "platform strategy",
    "project as practices": "project-as-practice",
    "project designing": "project design",
    "peer led team learning": "Peer-Led Team Learning (PLTL)",
    "peer-lead team learning": "Peer-Led Team Learning (PLTL)",
    "school leadership  management & administration": "school leadership, management & administration",
    "science  technology  engineering and mathematical": "STEM",
    "quality 4 0": "Quality 4.0",
    "school leaders’ perceptions": "school leaders' perceptions",
    "school leader’ perception": "school leaders' perceptions",
    "readiness for digital transformation": "digital transformation readiness",
      # Autoconcepto / “self-*”
    "self awareness": "self-awareness",
    "self regulation": "self-regulation",
    "self presences": "self-presence",
    "selfleadership": "self-leadership",
    "self-managed virtual teams": "self-managed virtual teams",
    "self-managing teams": "self-managing teams",

    # Entrevistas y diseños
    "semi-structured interviews": "semi-structured interviews",
    "sequential explanatory design": "sequential explanatory design",

    # Servicios/industrias/softwares
    "service-oriented softwares": "service-oriented software",
    "service companies": "service firms",
    "service industry": "service industries",

    # Compartidos/colaboración
    "shared leadership models": "shared leadership",
    "shared leadership technological leadership": "shared leadership",
    "shared collaboration": "collaborative sharing",

    # Online/edtech
    "student's characteristics": "students’ characteristics",
    "students  medical": "medical students",
    "students  pharmacy": "pharmacy students",
    "students  public health": "public health students",
    "storytelling  digital storytelling": "digital storytelling",

    # Estrategia / práctica
    "strategy as practices": "strategy-as-practice",
    "strategic-position leadership": "strategic positioning in leadership",

    # Regiones / ortografía
    "south-east asia": "Southeast Asia",
    "south and central america": "South and Central America",

    # Modelos/abreviaturas
    "structural equation model (sem)": "Structural Equation Modeling (SEM)",
    "structural equation modeling (sem)": "Structural Equation Modeling (SEM)",
    "stepwise weight assessment ratio analysis": "SWARA",
    "step-wise weight assessment ratio analysis (swara)": "SWARA",
    "technology organization environment theory": "TOE framework",
    "toe": "TOE framework",
    "the unified theory of acceptance and use of technology(utaut)": "UTAUT (Unified Theory of Acceptance and Use of Technology)",
    "tqm practices": "TQM practices",
    "tqm implementation": "TQM implementation",
    "tpack": "TPACK",

    # Variantes/hífen
    "social-networking": "social networking",
    "social-exchange": "social exchange",
    "two stage games": "two-stage games",
    "vehicle's dynamics": "vehicle dynamics",
    "user-experience": "user experience",
    "value-based": "value-based leadership" ,  # si aparece aislado en liderazgo
    "virtual and augmented reality": "virtual and augmented reality (VR/AR)",

    # Limpiezas varias
    "simple random sampling": "simple random sampling",
    "single lens reflexes": "single-lens reflex",
    "smart hr 4 0": "Smart HR 4.0",
    "ubiqutious media": "ubiquitous media",
    "spss analysis": "SPSS analysis",
     # Variantes de "virtual leadership / teams"
    "virtual leader": "virtual leadership",
    "virtual leadership  e-leadership": "virtual leadership (e-leadership)",
    "virtual leadership or e-leadership": "virtual leadership (e-leadership)",
    "virtual leadership and virtual teams": "virtual leadership and virtual teams",
    "vt leadership": "virtual team leadership",
    "virtual teams' (vt) management": "virtual team management",
    "vts challenges": "virtual team challenges",
    "virtual project managements": "virtual project management",
    "virtual project team": "virtual project teams",

    # Comunidad/aprendizaje
    "virtual learning community": "virtual learning communities",
    "virtual learning environment": "Virtual Learning Environment (VLE)",
    "virtual professional leadership learning communities (vplcs)":
        "Virtual Professional Leadership Learning Communities (VPLCs)",
    "virtual professional leadership learning communities":
        "Virtual Professional Leadership Learning Communities (VPLCs)",

    # Reuniones/formación
    "virtual professional development": "Virtual Professional Development (VPD)",
    "virtual writing workshop": "virtual writing workshops",
    "virtual morning report": "virtual morning report",
    "virtual vs in-person": "virtual versus in-person",

    # Simulación/realidad virtual
    "virtual reality simulations": "VR simulations",
    "virtual simulation": "virtual simulations",
    "virtual simulation environments": "virtual simulations",

    # Tecnologías visuales/visión
    "visual simultaneous localization and mappings":
        "visual simultaneous localization and mapping (vSLAM)",
    "visual slam": "visual simultaneous localization and mapping (vSLAM)",
    "vision-based approaches": "vision-based approaches",

    # Web / plataformas
    "web 2": "Web 2.0",
    "web 2 0": "Web 2.0",
    "word-of-mouth": "Word of Mouth (WOM)",

    # VUCA
    "vuca world": "VUCA world",
    "vuca paradox": "VUCA paradox",

    # Wearables
    "wearables": "wearable technology",

    # Trabajo/vida/estrés/bienestar
    "work–family": "work–family",
    "work–family conflict": "work–family conflict",
    "work related wellbeing": "work-related well-being",
    "work engagements": "work engagement",
    "work-flows": "workflows",

    # Otros ajustes ortográficos
    "virtual project team": "virtual project teams",
    "virtual teaching": "virtual instruction",
    "virtual office": "virtual offices",
    "virtual organization": "virtual organizations",


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
#df['bothKeywords'] = reemplazar_parciales(df['bothKeywords'], patrones_parciales)
df['Index Keywords'] = reemplazar_parciales(df['Index Keywords'], patrones_parciales)
df['Author Keywords'] = reemplazar_parciales(df['Author Keywords'], patrones_parciales)
# Guardar el DataFrame modificado en un nuevo archivo CSV
#df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)

df.to_csv(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replace.csv", index=False)
print("Palabras clave reemplazadas y nuevo archivo guardado.")