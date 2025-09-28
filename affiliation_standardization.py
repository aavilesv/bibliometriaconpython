import pandas as pd
import re
from rapidfuzz import fuzz, process
#este sirve 
# 1) Leer CSV

# numero final
ruta = r"G:/Mi unidad/Artículos cientificos/articulo 1/_affil_org_countryrevisarr.csv"
OUT =r"G:/Mi unidad/Artículos cientificos/articulo 1/_affil_org_countryfinalizar2.csv"
#numero 4 para buscar data

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
df['Affiliations_final'] = df['Affiliations_final'].apply(normalize_cell)
df['Authors with affiliations_final'] = df['Authors with affiliations_final'].apply(normalize_cell)



# ================== COLUMNAS ==================
COL_A = "Affiliations_final"
COL_B = "Authors with affiliations_final"

# ================== UMBRAL DE-DUPE (0 desactiva fuzzy) ==================
DEDUPE_THRESHOLD = 98  # 70-85 suele ir bien; más alto = más estricto

# ================== PRIORIDAD (de mayor a menor) ==================
PRIORITY_PATTERNS = [
    re.compile(r"\buniv\w*", re.IGNORECASE),                      # universidad / university / université / università ...
    re.compile(r"\binst\w*", re.IGNORECASE),                      # institute / instituto / institution ...
    re.compile(r"\bescue\w*", re.IGNORECASE),                     # escuela (castellano)

    re.compile(r"^\s*school\w*", re.IGNORECASE),                          # school
    re.compile(r"^\s*college\b", re.IGNORECASE),                         # college
    re.compile(r"^\s*acad(?:emia\w*|emy\w*)", re.IGNORECASE),            # academia / academy
    re.compile(r"^\s*fac(?:ultad\w*|ulty\w*)", re.IGNORECASE),           # facultad / faculty
    re.compile(r"^\s*muse(?:o|um)\w*", re.IGNORECASE),                   # museo / museum
    re.compile(r"^\s*hosp\w*", re.IGNORECASE),                           # hospital/hôpital
    re.compile(r"^\s*cent(?:er|re|ro|rum)?\w*", re.IGNORECASE),          # center/centre/centro/centrum
    re.compile(r"^\s*clin\w*", re.IGNORECASE),                           # clínica/clinic
    re.compile(r"^\s*minist(?:erio\w*|ry\w*)", re.IGNORECASE),           # ministerio / ministry
    re.compile(r"^\s*lab\w*", re.IGNORECASE),                            # laboratorio/lab
    re.compile(r"^\s*observ\w*", re.IGNORECASE),                         # observatorio/observatory
    re.compile(r"^\s*fund(?:acion\w*|aci[oó]n\w*|ation\w*)", re.IGNORECASE),  # fundación / foundation
    re.compile(r"^\s*corp\w*", re.IGNORECASE),                           # corporación/corporation
    re.compile(r"^\s*gov\w*", re.IGNORECASE),                            # gobierno/government
    re.compile(r"^\s*auth\w*", re.IGNORECASE),                           # autoridad/authority
    re.compile(r"^\s*cons\w*", re.IGNORECASE),                           # consejo/council
    re.compile(r"^\s*serv\w*", re.IGNORECASE),                           # servicio/service
    re.compile(r"^\s*(?:depart\w*|dept\b|dep\b)", re.IGNORECASE),        # departamento/department
    re.compile(r"^\s*flac\w*", re.IGNORECASE),                           # FLACSO        <-- antes estaba mal como 'sflac'
    re.compile(r"^\s*investig\w*", re.IGNORECASE),                       # investig-     <-- antes estaba mal como 'sinvestigador'
]

def pick_primary_org(fragment: str) -> str | None:
    """
    De un fragmento (texto entre ';'), parte por comas y regresa
    el PRIMER segmento que cumpla la prioridad (univ > inst > ...).
    No modifica idioma ni abreviaturas.
    """
    if not fragment or not str(fragment).strip():
        return None
    segments = [s.strip() for s in str(fragment).split(',') if s.strip()]
    if not segments:
        return None
    for rx in PRIORITY_PATTERNS:
        for seg in segments:
            if rx.search(seg):
                return seg
    return None

def dedupe_fuzzy(items: list[str], threshold: int) -> list[str]:
    """
    De-duplica preservando orden. Si threshold > 0, usa fuzzy (token_set_ratio).
    Si dos items matchean >= threshold, se queda el más largo/informativo.
    """
    out: list[str] = []
    if threshold <= 0:
        seen = set()
        for it in items:
            key = it.lower()
            if key not in seen:
                out.append(it); seen.add(key)
        return out

    for cand in items:
        if not out:
            out.append(cand); continue
        best = process.extractOne(cand, out, scorer=fuzz.token_set_ratio)
        if best and best[1] >= threshold:
            match_str, score, idx = best
            # conservar el más largo (más informativo)
            if len(cand) > len(match_str):
                out[idx] = cand
        else:
            # además evita duplicado exacto-insensible
            if cand.lower() not in (x.lower() for x in out):
                out.append(cand)
    return out

def merge_and_pick_orgs(a: str, b: str, threshold: int) -> str:
    """
    Une A+B, parte por ';', aplica pick_primary_org a cada fragmento,
    deduplica (con umbral), y retorna una sola cadena separada por '; '.
    """
    all_frags = []
    for source in (a or "", b or ""):
        all_frags.extend([x.strip() for x in str(source).split(';') if x.strip()])

    picked = []
    for frag in all_frags:
        org = pick_primary_org(frag)
        if org:
            picked.append(org)

    deduped = dedupe_fuzzy(picked, threshold)
    return '; '.join(deduped)

# ================== MAIN ==================


faltan = {COL_A, COL_B} - set(df.columns)
if faltan:
    raise ValueError(f"❌ Faltan columnas esperadas: {faltan}")




df["Combined_universities"] = df.apply(
    lambda r: merge_and_pick_orgs(r.get(COL_A, ""), r.get(COL_B, ""), DEDUPE_THRESHOLD),
    axis=1
)

# Métrica rápida
total = len(df)
vacias = int((df["Combined_universities"].str.strip() == "").sum())
print(f"Filas: {total:,} | Vacías en Combined_universities: {vacias:,} ({vacias/max(total,1):.2%})")
#df["Authors with affiliations"] = df["Combined_universities"] 
#df["Affiliations"] = df["Combined_universities"] 
#df = df.drop(columns=['Combined_universities'])
df.to_csv(OUT, index=False, encoding="utf-8")
print(f"📄 Guardado: {OUT}")
