import pandas as pd
import re,ast
from rapidfuzz import fuzz, process

# 1) Leer CSV
ruta =r"G:\\Mi unidad\\Master en administración y empresas\\articulo 3\\data\\datawos_scopus.csv"

df = pd.read_csv(ruta).fillna("")

# 2) Lista de países y diccionario de sinónimos
countries = [
   "Algeria", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Canada", "China",
    "Colombia", "Costa Rica", "Czech Republic", "Denmark", "Egypt", "Chile", 
    "Ethiopia", "Finland", "France", "Germany", "Ghana",
    "Greece", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Lebanon",
    "Lithuania", "Malaysia", "Mexico", "Morocco", "Nepal", "Netherlands",
    "New Zealand", "Nigeria", "Norway", "Oman", "Pakistan", "Palestine",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
    "Russia", "Rwanda", "Saudi Arabia", "Senegal", "Serbia", "Singapore",
    "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka",
    "Sweden", "Switzerland", "Taiwan", "Thailand", "Tunisia", "Turkey",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Uzbekistan", "Vietnam", "Zimbabwe", "Ivory Coast", "Kuwait",
    "Croatia", "Afghanistan", "Albania", "Andorra", "Angola", "Armenia",
    "Azerbaijan", "Bahamas", "Bahrain", "Barbados", "Belarus", "Belize",
    "Benin", "Bhutan", "Bolivia", "Botswana", "Brunei Darussalam",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon",
    "Central African Republic", "Chad", "Comoros", "Congo", "Cuba",
    "Cyprus", "Djibouti", "Dominican Republic", "Ecuador", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Fiji", "Gabon",
    "Gambia", "Georgia", "Grenada", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Iceland", "Jamaica", "Kiribati", "Kyrgyzstan", "Laos",
    "Latvia", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Luxembourg",
    "Madagascar", "Malawi", "Maldives", "Mali", "Malta", "Marshall Islands",
    "Mauritania", "Mauritius", "Micronesia", "Monaco", "Mongolia",
    "Montenegro", "Mozambique", "Myanmar", "Namibia", "Nauru", "Niger",
    "North Macedonia", "Palau", "Panama", "Papua New Guinea", "Paraguay",
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Seychelles", "Sierra Leone", "Slovakia",
    "Solomon Islands", "Somalia", "South Sudan", "Sudan", "Suriname",
    "Tajikistan", "Tanzania", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Turkmenistan", "Tuvalu", "Uganda", "Uruguay",
    "Vanuatu", "Venezuela", "Yemen", "Zambia", "Swaziland"
]


synonyms_map = {c.lower(): c for c in countries}



# 3) Compilar patrones
country_vals = '|'.join(map(re.escape, synonyms_map.values()))
# Detecta comas mal puestas tras un país, independientemente de siguiente palabra
sep_pattern = re.compile(rf'({country_vals})\s*,\s*(?=[A-Z])')
# Patron de contenido entre corchetes
bracket_pattern = re.compile(r'\[.*?\]')

def normalize_cell(raw: str) -> str:
    # 1) Eliminar contenido entre corchetes
    text = bracket_pattern.sub('', raw)
    # 2) Reemplazar comas tras país por ';'
    text = sep_pattern.sub(r'\1; ', text)
    # 3) Split en fragments
    frags = [frag.strip() for frag in text.split(';') if frag.strip()]
    # 4) Asegurar país al final de cada fragmento
    norm = []
    for i, frag in enumerate(frags):
        # Si termina en un país canónico, OK
        if any(frag.endswith(c) for c in synonyms_map.values()):
            norm.append(frag)
        else:
            # Intentar tomar país de siguiente fragmento
            if i + 1 < len(frags):
                for key, val in synonyms_map.items():
                    if frags[i+1].lower().startswith(val.lower()):
                        norm.append(f"{frag}, {val}")
                        break
                else:
                    norm.append(frag)
            else:
                norm.append(frag)
    # 5) Unir de nuevo
    return '; '.join(norm)

# 4) Aplicar a columnas
df['Affiliations'] = df['Affiliations'].apply(normalize_cell)
df['Authors with affiliations'] = df['Authors with affiliations'].apply(normalize_cell)

keys = [  'univ',    # University / Universidad
  'inst',    # Institute / Instituto / Institution
  'escu',    # Escuela (castellano),
  'scho',    # School / Escuela técnica
  'coll',    # College
   'acad',   # Academy / Academia
  'fac',     # Faculty / Facultad
  'hos',      # Hospital
  'center', #centro
  'minist',  # ministerio
  'depart' #departamento
]

# 3) Función para extraer parte principal de un fragmento
def extract_from_frag(frag):
    for key in keys:
        if re.search(key, frag, re.IGNORECASE):
            parts = [p.strip() for p in frag.split(',') if p.strip()]
            # segmento que contiene la palabra clave
            inst_part = next((p for p in parts if re.search(key, p, re.IGNORECASE)), parts[0])
            # verificar si el último segmento es un país
            last = parts[-1].strip().lower()
            country_part = synonyms_map.get(last)
            if country_part:
                return f"{inst_part}, {country_part}"
            return inst_part
    return None

# 4) Función para procesar toda la celda
def extract_primary(cell):
    frags = [f.strip() for f in cell.split(';') if f.strip()]
    extracted = [extract_from_frag(f) for f in frags]
    # filtrar None y unir
    return '; '.join([e for e in extracted if e])

# 5) Aplicar a ambas columnas
df['Affiliations'] = df['Affiliations'].apply(extract_primary)
df['Authors with affiliations'] = df['Authors with affiliations'].apply(extract_primary)


# ---------- 2) Abreviaturas a expandir ----------
ABBR_EXPANSIONS = {
    r'\bUniv\b':  'University',
    r'\bInst\b':  'Institute',
    r'\bColl\b':  'College',
    r'\bSch\b':   'School',
    r'\bFac\b':   'Faculty',
    r'\bAcad\b':  'Academy',
    r'\bDept\b':  'Department',
    r'\bEscu\b':  'Escuela',
    r'\bMinis\b':  'Ministry',
    
    
}

def expand_abbrev(text: str) -> str:
    """Expande las abreviaturas definidas en ABBR_EXPANSIONS."""
    for pat, repl in ABBR_EXPANSIONS.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text.strip()

# ---------- 3) Palabras clave jerárquicas opcionales ----------
KEY_PAT = re.compile(
    r'\b(' + '|'.join([
        'univ', 'inst', 'acad', 'fac', 'coll',
        'scho', 'escu', 'hos', 'center', 'minist', 'depart'
    ]) + r')', flags=re.IGNORECASE
)


# -------------- División en (inst, country) --------------
def split_inst_country(segment: str, country_map: dict[str, str]):
    inst, _, tail = segment.rpartition(',')
    inst, tail = inst.strip(), tail.strip()
    country = country_map.get(tail.lower(), '')
    if inst and country:
        return inst, country
    return None           # marca para descartar

# -------------- Función principal con seguimiento -------
def combine_affils_with_removed(a_str: str,
                                b_str: str,
                                country_map: dict[str, str],
                                threshold: int = 70):
    """
    Devuelve:
      · combined_str  – afiliaciones deduplicadas 'Inst, País; ...'
      · removed_list  – lista de fragmentos descartados
    """
    removed = list()          # aquí guardaremos lo eliminado

    def preprocess(source):
        frags = []
        for seg in source.split(';'):
            seg_exp = expand_abbrev(seg.strip())
            if not seg_exp:
                continue
            if not KEY_PAT.search(seg_exp):
                removed.append(seg_exp)        # sin palabra clave
                continue
            res = split_inst_country(seg_exp, country_map)
            if res:
                frags.append(res)
            else:
                removed.append(seg_exp)        # sin país válido
        return frags

    list_a = preprocess(a_str)
    list_b = preprocess(b_str)
    combined = list(list_a)

    # --------------- Deduplicación (con versión más larga) ---------------
    for inst_b, ctry_b in list_b:
        candidates = [inst for inst, _ in combined]
        best = process.extractOne(inst_b, candidates, scorer=fuzz.token_set_ratio)
        if best and best[1] >= threshold:
            match_inst, score, idx = best
            longer = inst_b if len(inst_b) > len(match_inst) else match_inst
            combined[idx] = (longer, combined[idx][1])
            continue
        best2 = process.extractOne(inst_b, candidates, scorer=fuzz.partial_ratio)
        if best2 and best2[1] >= threshold - 10:
            match_inst2, score2, idx2 = best2
            longer = inst_b if len(inst_b) > len(match_inst2) else match_inst2
            combined[idx2] = (longer, combined[idx2][1])
            continue
        combined.append((inst_b, ctry_b))

    # --------------- Reconstruir cadena final ----------------
    seen, out = set(), []
    for inst, ctry in combined:
        item = f"{inst}, {ctry}"
        if item not in seen:
            out.append(item)
            seen.add(item)
    combined_str = '; '.join(out)
    return combined_str, removed


# country_map = {c.lower(): c for c in countries}
df[['Combined_affiliations', 'Removed_affiliations']] = df.apply(
    lambda row: pd.Series(
        combine_affils_with_removed(
            row['Affiliations'],
            row['Authors with affiliations'],
            synonyms_map,
            threshold=80
        )
    ),
    axis=1
)

## revisar que organzaciones están en el países
MANUAL_COUNTRY = {
    
    "allameh tabatabai university": "Iran",
    "arkansas state university": "United States",
    "academy of romanian scientists": "Romania",
    "alliance of bioversity international and the international center for tropical agriculture (ciat)": "Colombia",
    "charles darwin university": "Australia",
    "agricultural university of athens": "Greece",
    "chalmers university technol": "Sweden",
    "charles university": "Czech Republic",
    "british council learning center 0108": "United States",
    "media & creative industries department united arab emirates university uae, jordan": "United Arab Emirates",
    "indiana university system, jordan":"United States",
    "girne american university":"Cyprus",
        "german archaeological institute":"Germany",
        "chandigarh university": "India",
        "loughborough university": "United Kingdom",
        "concordia university, bahrain": "Canada", # Requiere verificación adicional
        "luleå tekniska universitet": "Sweden",



 

    # … añade los que necesites
}
MANUAL_COUNTRY = {k.lower(): v for k, v in MANUAL_COUNTRY.items()}

# patrón con todos los países
COUNTRY_PAT = re.compile(r'\b(' + '|'.join(map(re.escape, countries)) + r')\b',
                         re.IGNORECASE)


def normalize_country(seg: str) -> str | None:
    """Devuelve 'Inst, País' o None si no se puede asignar país."""
    seg = expand_abbrev(seg.strip().strip("'\""))
    if not seg:
        return None

    # a)  BUSCAR en el diccionario manual
    for key, ctry in MANUAL_COUNTRY.items():
        if key in seg.lower():
            return f"{seg}, {ctry}"

    # b)  ¿Ya termina en ', País'?
    inst, _, tail = seg.rpartition(',')
    if tail.strip().lower() in synonyms_map:
        return f"{inst.strip()}, {synonyms_map[tail.strip().lower()]}"

    # c)  Buscar el nombre del país dentro del texto
    m = COUNTRY_PAT.search(seg)
    if m:
        country = synonyms_map[m.group(1).lower()]
        return f"{seg}, {country}"

    # d)  No se pudo determinar
    return None
def clean_removed_cell(cell: str, threshold: int = 70) -> str:
    """
    Converte la celda '[... ]' en texto plano deduplicado 'Inst, País; …'
    """
    if not cell or cell == "[]":
        return ""
    try:
        items = ast.literal_eval(cell) if cell.startswith('[') else [cell]
    except Exception:
        items = [cell]

    # 1) normalizar y descartar los que sigan sin país
    tuples = []
    for itm in items:
        norm = normalize_country(itm)
        if norm:
            inst, _ , country = norm.rpartition(',')
            tuples.append((inst.strip(), country.strip()))

    # 2) deduplicar (token_set_ratio)
    uniques = []
    for inst, ctry in tuples:
        if not uniques:
            uniques.append((inst, ctry)); continue
        cand = [u[0] for u in uniques]
        best = process.extractOne(inst, cand, scorer=fuzz.token_set_ratio)
        if best and best[1] >= threshold:
            idx = best[2]
            longer = inst if len(inst) > len(cand[idx]) else cand[idx]
            uniques[idx] = (longer, ctry)
        else:
            uniques.append((inst, ctry))

    return '; '.join(f"{i}, {c}" for i, c in uniques)
df['Removed_affiliations_clean'] = (
    df['Removed_affiliations']
      .astype(str)                 # por si vienen listas como string
      .apply(clean_removed_cell)
)

def merge_affil_cols(combined: str, removed: str) -> str:
    """
    Une 'Combined_affiliations' y 'Removed_affiliations_clean'
    sin repetir 'Inst, País'.
    """
    items = []
    seen  = set()

    # 1) añadir los del combined (ya deduplicados)
    for seg in [s.strip() for s in combined.split(';') if s.strip()]:
        if seg not in seen:
            items.append(seg)
            seen.add(seg)

    # 2) añadir los del removed (ya limpios)
    for seg in [s.strip() for s in removed.split(';') if s.strip()]:
        if seg not in seen:
            items.append(seg)
            seen.add(seg)

    return '; '.join(items)

# Crear la nueva columna
df['All_affiliations'] = df.apply(
    lambda row: merge_affil_cols(
        row['Combined_affiliations'],
        row['Removed_affiliations_clean']
    ),
    axis=1
)


MANUAL_REPLACE = {
    # ——— Estados Unidos ———
    
    
    "academy romanian scientists"         : "academy of romanian scientists",
    "adam mickiewicz university" :"adam mickiewicz university in poznań",
    "adam mickiewicz university in poznań" :"adam mickiewicz university in poznań",
    "agh university of science and technology stanisław staszic in krakow" :"agh university of science and technology",
    "& lebanese american university	" :"lebanese american university",
    "“jožef stefan” institute" :"jožef stefan institute",
   "arizona state university tempe": "arizona state university",
    "arizona state university": "arizona state university",
    "australian national university": "australian national university",
    "australian natl university": "australian national university",
    "azerbaijan state pedag university": "azerbaijan state pedagogical university",
    "azerbaijan state pedagogical university": "azerbaijan state pedagogical university",
    "beihang university of china": "beihang university",
    "beihang university": "beihang university",
    "bentley college": "bentley university",
    "bentley university": "bentley university",
    "bi norwegian business school": "bi norwegian business school",
    "bi the norwegian business school": "bi norwegian business school",
    "brandenburg tech university cottbus": "brandenburg university of technology cottbus-senftenberg",
    "brandenburg university of technology cottbus-senftenberg": "brandenburg university of technology cottbus-senftenberg",
    "cent university finance & econ": "central university of finance and economics",
    "central university of finance and economics": "central university of finance and economics",
    "chalmers university of technology": "chalmers university of technology",
    "chalmers university technol": "chalmers university of technology",
    "chongqing university": "chongqing university",
    "chongqing university.": "chongqing university",
    "comenius university bratislava": "comenius university in bratislava",
    "comenius university in bratislava": "comenius university in bratislava",
    "comsats institute of information technology": "comsats university islamabad",
    "comsats university islamabad": "comsats university islamabad",
    "concordia university - canada": "concordia university",
    "concordia university": "concordia university",
    "cranfield school of management": "cranfield university",
    "cranfield university": "cranfield university",
    "deakin university in victoria": "deakin university",
    "deakin university": "deakin university",
    "delhi technol university": "delhi technological university",
    "delhi technological university": "delhi technological university",
    "b. s. abdur rahman university": "b.s. abdur rahman crescent institute of science & technology / university",
    "department of commerce b.s. abdur rahman crescent institute of science & technology": "b.s. abdur rahman crescent institute of science & technology / university",
    "ecampus university": "ecampus university",
    "e-campus university": "ecampus university",
    "edith cowan university and sir charles gairdner osborne park health care group": "edith cowan university",
    "edith cowan university": "edith cowan university",
    "erasmus university rotterdam": "erasmus university rotterdam",
    "erasmus university": "erasmus university rotterdam",
    "federico ii university naples": "federico ii university of naples",
    "federico ii university of naples": "federico ii university of naples",
    "fundacion pérez scremini-hospital pereira rossell": "fundación pérez scremini-hospital pereira rossell",
    "fundación pérez scremini-hospital pereira rossell": "fundación pérez scremini-hospital pereira rossell",
    "gdansk university of technology": "gdańsk university of technology",
    "gdańsk university of technology": "gdańsk university of technology",
    "georg august university goettingen": "georg-august-universität göttingen",
    "georg-august-universität göttingen": "georg-august-universität göttingen",
    "government college of management sciences": "government college of management sciences",
    "govt college management sci": "government college of management sciences",
    "harbin institute of technology (weihai)": "harbin institute of technology",
    "harbin institute of technology": "harbin institute of technology",
    "harbin institute technol": "harbin institute of technology",
    "harbin university of science and technology": "harbin university of science and technology",
    "harbin university sci & technol": "harbin university of science and technology",
    "heriot watt university": "heriot-watt university",
    "huazhong agricultural university": "huazhong agricultural university",
    "huazhong agricultural university.": "huazhong agricultural university",
    "humboldt university of berlin": "humboldt university of berlin",
    "humboldt university": "humboldt university of berlin",
    "humboldt-universität zu berlin": "humboldt university of berlin",
    "indian institute of management (iim system)": "indian institute of management (iim system)",
    "indian institute of management amritsar": "indian institute of management amritsar",
    "indian institute of management lucknow": "indian institute of management lucknow",
    "indonesian collage of tourism economic (stiepari)": "indonesian college of tourism economics (stiepari)",
    "master management. indonesian collage of tourism economic (stiepari)": "indonesian college of tourism economics (stiepari)",
    "institute for advanced sustainability studies e.v. (iass)": "institute for advanced sustainability studies (iass)",
    "institute for advanced sustainability studies": "institute for advanced sustainability studies (iass)",
    "institute for research and training in agriculture and fisheries (ifapa)": "institute for research and training in agriculture and fisheries (ifapa)",
    "institute of agricultural and fisheries research and training (ifapa)": "institute for research and training in agriculture and fisheries (ifapa)",
    "instituto de investigación y formación agraria y pesquera (ifapa)": "institute for research and training in agriculture and fisheries (ifapa)",
    "college of optometrists": "college of optometrists / institute of optometry",
    "institute of optometry": "college of optometrists / institute of optometry",
    "institute of science & technology - austria": "institute of science and technology austria (ista)",
    "institute sci & technol austria ista": "institute of science and technology austria (ista)",
    "international center for tropical agriculture ciat": "international center for tropical agriculture (ciat)",
    "international center for tropical agriculture": "international center for tropical agriculture (ciat)",
    "iscte-instituto universitário de lisboa (iscte-iul)": "iscte - instituto universitário de lisboa",
    "instituto universitário de lisboa": "iscte - instituto universitário de lisboa",
    "jacobs university bremen": "jacobs university bremen",
    "jacobs university": "jacobs university bremen",
    "jiangmen industrial technology research institute co.": "jiangmen industrial technology research institute",
    "jiangmen industrial technology research institute of guangdong academy of sciences": "jiangmen industrial technology research institute",
    "kings college london": "king's college london",
    "king’s college london": "king's college london",
    "king’s business school": "king's college london",
    "brazil laboratório de justiça territorial da universidade federal do abc": "laboratório de justiça territorial/ambiental da universidade federal do abc",
    "laboratório de justiça ambiental da universidade federal do abc": "laboratório de justiça territorial/ambiental da universidade federal do abc",
    "laboratório de justiça territorial da universidade federal do abc": "laboratório de justiça territorial/ambiental da universidade federal do abc",
    "& lebanese american university": "lebanese american university",
    "lebanese american university": "lebanese american university",
    "leuphana universität lüneburg": "leuphana university lüneburg",
    "leuphana university luneburg": "leuphana university lüneburg",
    "leuphana university lüneburg": "leuphana university lüneburg",
    "leuphana university": "leuphana university lüneburg",
    "linkoping university": "linköping university",
    "linköping university": "linköping university",
    "mcgill university health center": "mcgill university health centre",
    "mcgill university health centre": "mcgill university health centre",
    "doctor management education. merchant marine collage": "merchant marine college",
    "masterof law. merchant marine collage": "merchant marine college",
    "middle east tech university": "middle east technical university (metu)",
    "middle east technical university cankaya": "middle east technical university (metu)",
    "middle east technical university": "middle east technical university (metu)",
    "hse university": "national research university higher school of economics (hse university)",
    "national research university higher school of economics (hse university)": "national research university higher school of economics (hse university)",
    "natl res university": "national research university higher school of economics (hse university)",
    "national school for political and administrative studies": "national university of political studies and public administration (snspa)",
    "national university of political studies & public administration (snspa) - romania": "national university of political studies and public administration (snspa)",
    "national university of political studies and public administration": "national university of political studies and public administration (snspa)",
    "natl university polit studies & publ adm snspa": "national university of political studies and public administration (snspa)",
    "natl university polit studies & publ adm": "national university of political studies and public administration (snspa)",
    "nelson mandela university (nmu)": "nelson mandela university (nmu)",
    "nelson mandela university": "nelson mandela university (nmu)",
    "northeast forestry university - china": "northeast forestry university",
    "northeast forestry university": "northeast forestry university",
    "norwegian university of science and technology": "norwegian university of science and technology (ntnu)",
    "norwegian university sci & technol ntnu": "norwegian university of science and technology (ntnu)",
    "o p jindal global university": "o.p. jindal global university",
    "o.p. jindal global university": "o.p. jindal global university",
    "open university - united kingdom": "the open university",
    "open university": "the open university"
    
    
    
    
    # ——— Añade los que vayas detectando ———
}
# Compilar regex con todas las claves (ignora mayúsculas/minúsculas)
pattern = re.compile('|'.join(map(re.escape, MANUAL_REPLACE.keys())), re.IGNORECASE)

def replace_manual(cell: str) -> str:
    """
    Sustituye cada variante por su forma canónica, respetando el orden original.
    """
    def repl(match):
        original = match.group(0)
        return MANUAL_REPLACE.get(original.lower(), original)   # por si cambia gist
    return pattern.sub(repl, cell)

# Aplicar sobre la columna ya limpia (All_affiliations o la que quieras)
df['All_affiliations_norm'] = df['All_affiliations'].apply(replace_manual)
df['Affiliations']  = df['All_affiliations_norm'] 
df['Authors with affiliations']  = df['All_affiliations_norm'] 
#  drop() eliminar
df = df.drop('Combined_affiliations', axis=1)  # axis=1 para columnas
df = df.drop('Removed_affiliations', axis=1)  # axis=1 para columnas
df = df.drop('Removed_affiliations_clean', axis=1)  # axis=1 para columnas
df = df.drop('All_affiliations', axis=1)  # axis=1 para columnas
df = df.drop('All_affiliations_norm', axis=1)  # axis=1 para columnas


# 9) Guarda el resultado
out = r"G:\\Mi unidad\\Master en administración y empresas\\articulo 3\\data\\datawos_scopuafliacion.csv"

df.to_csv(out, index=False)
print("Resultado guardado en:", out)


