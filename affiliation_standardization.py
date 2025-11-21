import pandas as pd
import re
from rapidfuzz import fuzz, process
from typing import Optional   # 👈 Import necesario

# 1) Leer CSV

ruta =  r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusbloque1replacelematizar.csv"
OUT =  r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusafiliation.csv"
# Forzar a string para evitar DtypeWarning
df = pd.read_csv(ruta, low_memory=False, dtype=str).fillna("")

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
sep_pattern = re.compile(rf'({country_vals})\s*,\s*(?=[A-Z])')  # comas mal puestas
bracket_pattern = re.compile(r'\[.*?\]')  # contenido entre corchetes

def normalize_cell(raw: str) -> str:
    text = bracket_pattern.sub('', raw)
    text = sep_pattern.sub(r'\1; ', text)
    frags = [frag.strip() for frag in text.split(';') if frag.strip()]
    norm = []
    for i, frag in enumerate(frags):
        if any(frag.endswith(c) for c in synonyms_map.values()):
            norm.append(frag)
        else:
            if i + 1 < len(frags):
                for key, val in synonyms_map.items():
                    if frags[i+1].lower().startswith(val.lower()):
                        norm.append(f"{frag}, {val}")
                        break
                else:
                    norm.append(frag)
            else:
                norm.append(frag)
    return '; '.join(norm)

# Aplicar normalización
df['Affiliations'] = df['Affiliations'].apply(normalize_cell)
df['Authors with affiliations'] = df['Authors with affiliations'].apply(normalize_cell)

# ================== COLUMNAS ==================
COL_A = "Affiliations"
COL_B = "Authors with affiliations"

# ================== UMBRAL DE-DUPE ==================
DEDUPE_THRESHOLD = 98

# ================== PRIORIDAD ==================
PRIORITY_PATTERNS = [
    re.compile(r"\buniv\w*", re.IGNORECASE),
    re.compile(r"\binst\w*", re.IGNORECASE),
    re.compile(r"\bescue\w*", re.IGNORECASE),
    re.compile(r"^\s*school\w*", re.IGNORECASE),
    re.compile(r"^\s*college\b", re.IGNORECASE),
    re.compile(r"^\s*acad(?:emia\w*|emy\w*)", re.IGNORECASE),

    re.compile(r"^\s*minist(?:erio\w*|ry\w*)", re.IGNORECASE),

    re.compile(r"^\s*fund(?:acion\w*|aci[oó]n\w*|ation\w*)", re.IGNORECASE),
    re.compile(r"^\s*corp\w*", re.IGNORECASE),
]

def pick_primary_org(fragment: str) -> Optional[str]:   # 👈 corregido
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
            if len(cand) > len(match_str):
                out[idx] = cand
        else:
            if cand.lower() not in (x.lower() for x in out):
                out.append(cand)
    return out

def merge_and_pick_orgs(a: str, b: str, threshold: int) -> str:
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
df["Authors with affiliations"] = df["Affiliations"]
df.drop(columns=['Combined_universities'], inplace=True)
df.to_csv(OUT, index=False, encoding="utf-8")
print(f"📄 Guardado: {OUT}")
