import pandas as pd
import re
from rapidfuzz import fuzz, process

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
ruta = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusbloque1replacelematizar.csv"
OUT = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusafiliation.csv"

# Cargar datos
df = pd.read_csv(ruta, low_memory=False, dtype=str).fillna("")

# ==========================================
# 2. DICCIONARIOS DE TRANSFORMACIÓN
# ==========================================

# A) PARA COMPARAR (Expandimos para que el Fuzzy Match sea preciso)
ABBREV_MAP_EXPAND = {
    r"\bUniv\b": "University",
    r"\bInst\b": "Institute",
    r"\bAcad\b": "Academy",
    r"\bSci\b": "Sciences",
    r"\bTech\b": "Technology",
    r"\bEng\b": "Engineering",
    r"\bDept\b": "Department",
    r"\bLab\b": "Laboratory",
    r"\bNatl\b": "National",
    r"\bInt\b": "International",
    r"\bRes\b": "Research",
    r"\bCtr\b": "Center",
    r"\bCent\b": "Center",
    r"\bColl\b": "College",
    r"\bSch\b": "School",
    r"\bMinist\b": "Ministry",
}

# B) PARA GUARDAR (Comprimimos al final para ahorrar espacio)
# Aquí definimos cómo quieres que quede escrito en el Excel
COMPRESS_MAP_FINAL = {
    r"\bUniversity\b": "Univ",
    r"\bUniversit[àäa]\b": "Univ",   # Variantes de idiomas
    r"\bUniversidad\b": "Univ",      # Español
    r"\bInstitute\b": "Inst",
    r"\bInstituto\b": "Inst",
    r"\bAcademy\b": "Acad",
    r"\bDepartment\b": "Dept",
    r"\bDepartamento\b": "Dept",
    r"\bLaboratory\b": "Lab",
    r"\bLaboratorio\b": "Lab",
    r"\bNational\b": "Natl",
    r"\bInternational\b": "Int",
    r"\bResearch\b": "Res",
    r"\bCenter\b": "Ctr",
    r"\bCentre\b": "Ctr",
    r"\bCentro\b": "Ctr",
    r"\bCollege\b": "Coll",
    r"\bSchool\b": "Sch",
    r"\bEscuela\b": "Sch",
    r"\bTechnology\b": "Tech",
    r"\bSciences\b": "Sci",
    r"\bEngineering\b": "Eng",
    r"\bManagement\b": "Mgmt",
}

# Patrones de Organización Válida
VALID_ORG_PATTERNS = [
    re.compile(r"\buniv\w*", re.IGNORECASE),
    re.compile(r"\binst\w*", re.IGNORECASE),
    re.compile(r"\bminist\w*", re.IGNORECASE),
    re.compile(r"\bcourt\w*", re.IGNORECASE),
    re.compile(r"\btribunal\w*", re.IGNORECASE),
    re.compile(r"\bschool\w*", re.IGNORECASE),
    re.compile(r"\bfacul\w*", re.IGNORECASE),
    re.compile(r"\bcolleg\w*", re.IGNORECASE),
    re.compile(r"\bacad\w*", re.IGNORECASE),
    re.compile(r"\bcouncil\w*", re.IGNORECASE),
    re.compile(r"\bcommis\w*", re.IGNORECASE),
    re.compile(r"\bagenc\w*", re.IGNORECASE),
    re.compile(r"\bcent(?:er|re|ro|rum)\b", re.IGNORECASE),
    re.compile(r"\blab\w*", re.IGNORECASE),
    re.compile(r"\borg\w*", re.IGNORECASE),
    re.compile(r"\bassoc\w*", re.IGNORECASE),
    re.compile(r"\bpolitec\w*", re.IGNORECASE),
]

# Lista Negra
TRASH_PATTERNS = [
    re.compile(r"\bstreet\b", re.IGNORECASE),
    re.compile(r"\broad\b", re.IGNORECASE),
    re.compile(r"\bbox\b", re.IGNORECASE),
    re.compile(r"\bavenue\b", re.IGNORECASE),
    re.compile(r"@", re.IGNORECASE),
    re.compile(r"\bemail\b", re.IGNORECASE),
]

# Países y Ciudades
countries_list = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Canada", "China",
    "Chile", "Colombia", "Costa Rica", "Denmark", "Ecuador", "Egypt", "Finland", "France",
    "Germany", "Greece", "India", "Indonesia", "Iran", "Ireland", "Israel", "Italy", "Japan",
    "Kenya", "Malaysia", "Mexico", "Morocco", "Netherlands", "New Zealand", "Nigeria", "Norway",
    "Pakistan", "Peru", "Philippines", "Poland", "Portugal", "Russia", "Saudi Arabia",
    "Singapore", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Thailand",
    "Turkey", "Ukraine", "United Kingdom", "United States", "Vietnam", "Venezuela",
    "USA", "UK", "Russia", "Peoples R China"
]
COUNTRY_MAP = {c.lower(): c for c in countries_list}
COUNTRY_MAP.update({
    "usa": "United States", "uk": "United Kingdom", "peoples r china": "China", "pr china": "China"
})

CITY_TO_COUNTRY = {
    "new york": "United States", "washington": "United States", "boston": "United States",
    "london": "United Kingdom", "oxford": "United Kingdom", "cambridge": "United Kingdom",
    "paris": "France", "berlin": "Germany", "madrid": "Spain", "barcelona": "Spain",
    "rome": "Italy", "beijing": "China", "shanghai": "China", "wuhan": "China",
    "tokyo": "Japan", "seoul": "South Korea", "buenos aires": "Argentina", 
    "sao paulo": "Brazil", "brasilia": "Brazil", "santiago": "Chile", 
    "bogota": "Colombia", "lima": "Peru", "mexico city": "Mexico", "quito": "Ecuador",
    "canberra": "Australia", "sydney": "Australia", "melbourne": "Australia",
    "brussels": "Belgium", "geneva": "Switzerland", "moscow": "Russia", "amsterdam": "Netherlands",
    "wageningen": "Netherlands"
}

BRACKET_RE = re.compile(r'\[.*?\]')
SEPARATOR_NORMALIZER = re.compile(r'[\-\|\(\)\.]+') 

# ==========================================
# 3. FUNCIONES DE LIMPIEZA
# ==========================================

def expand_abbreviations(text: str) -> str:
    """Expande para comparar mejor (Univ -> University)"""
    for pattern, replacement in ABBREV_MAP_EXPAND.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def compress_final_name(text: str) -> str:
    """Comprime para guardar (University -> Univ)"""
    for pattern, replacement in COMPRESS_MAP_FINAL.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def infer_country(text: str) -> str:
    text_lower = text.lower()
    for c_name in sorted(COUNTRY_MAP.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(c_name)}\b", text_lower):
            return COUNTRY_MAP[c_name]
    for city, country in CITY_TO_COUNTRY.items():
        if re.search(rf"\b{re.escape(city)}\b", text_lower):
            return country
    return ""

def is_trash(text: str) -> bool:
    for trash_pat in TRASH_PATTERNS:
        if trash_pat.search(text): return True
    return False

def clean_org_name(org: str) -> str:
    org = re.sub(r'\d+', '', org) 
    return org.strip(" ,;-")

def process_affiliation_extraction(raw: str) -> list:
    if not raw or not isinstance(raw, str): return []

    clean = BRACKET_RE.sub('', raw)
    affiliations = [x.strip() for x in clean.split(';') if x.strip()]
    extracted_data = []
    
    for aff in affiliations:
        aff_normalized = SEPARATOR_NORMALIZER.sub(',', aff)
        parts = [p.strip() for p in aff_normalized.split(',') if p.strip()]
        
        found_country = infer_country(aff)
        if not found_country: continue 
            
        found_org = ""
        for part in parts:
            if is_trash(part): continue
            
            is_valid_org = False
            for valid_pat in VALID_ORG_PATTERNS:
                if valid_pat.search(part):
                    is_valid_org = True
                    break
            
            if is_valid_org:
                part_lower = part.lower()
                is_geo = (part_lower in COUNTRY_MAP) or (part_lower in CITY_TO_COUNTRY) or (part_lower == found_country.lower())
                if not is_geo:
                    found_org = part
                    break 
        
        if found_org:
            org_clean = clean_org_name(found_org)
            org_expanded = expand_abbreviations(org_clean)
            extracted_data.append({
                "expanded": org_expanded, # Usamos esta para comparar (Versión larga)
                "original": org_clean,    # Usamos esta para procesar (Versión origen)
                "country": found_country
            })

    return extracted_data

# ==========================================
# 4. DEDUPLICACIÓN Y COMPRESIÓN FINAL
# ==========================================

def deduplicate_affiliations(data_list: list) -> str:
    if not data_list: return ""
    
    unique_orgs = []
    
    for item in data_list:
        candidate = item
        candidate_text = candidate["expanded"]
        
        matched = False
        for i, existing in enumerate(unique_orgs):
            if candidate["country"] == existing["country"]:
                # Usamos 85 para equilibrar
                ratio = fuzz.token_set_ratio(candidate_text, existing["expanded"])
                if ratio > 85:
                    matched = True
                    # Nos quedamos con la versión "Original" más larga antes de comprimir
                    if len(candidate["original"]) > len(existing["original"]):
                        unique_orgs[i] = candidate
                    break
        
        if not matched:
            unique_orgs.append(candidate)
            
    # AQUÍ OCURRE LA MAGIA: Comprimimos los nombres seleccionados
    final_strings = []
    for item in unique_orgs:
        # Tomamos el nombre (sea cual sea que haya ganado) y lo comprimimos
        short_name = compress_final_name(item['original'])
        final_strings.append(f"{short_name}, {item['country']}")
        
    return "; ".join(final_strings)

# ==========================================
# 5. EJECUCIÓN
# ==========================================

print("⏳ Procesando: Extracción -> Deduplicación -> Estandarización (Univ)...")

df['Raw_Text'] = df['Affiliations'] + ";" + df['Authors with affiliations']
temp_extracted = df['Raw_Text'].apply(process_affiliation_extraction)
df['Combined_universities'] = temp_extracted.apply(deduplicate_affiliations)

total = len(df)
llenas = df['Combined_universities'].replace("", pd.NA).count()
print(f"✅ Filas válidas: {llenas} de {total}")

df["Affiliations"] = df["Combined_universities"]
df["Authors with affiliations"] = df["Combined_universities"]
df.drop(columns=['Combined_universities', 'Raw_Text'], inplace=True, errors='ignore')

df.to_csv(OUT, index=False, encoding="utf-8")
print(f"📄 Archivo guardado: {OUT}")