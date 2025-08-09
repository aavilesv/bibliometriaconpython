# -*- coding: utf-8 -*-
import re
import pandas as pd
from rapidfuzz import process, fuzz
from collections import Counter

# ================== CONFIG ==================
CSV_IN  = r"G:/Mi unidad/Artículos cientificos/articulo 1/datawos_scopus.csv"
CSV_OUT = CSV_IN.replace(".csv", "_affil_org_country4.csv")

COL_A = "Affiliations"
COL_B = "Authors with affiliations"

# ====== Keywords de instituciones ======
ORG_PRIMARY = [
    "university","universidad","universidade","college","institute","instituto","institution",
    "academy","academia","hospital","centre","center","centro","foundation","fundación",
    "ministry","ministerio","conicet","csic","cnrs","max planck","helmholtz",
    # italiano
    "università","istituto","museo","soprintendenza","politecnico","politecnica","cnr"
]
ORG_SECONDARY = [
    "faculty","facultad","school","escuela","department","departamento","laboratory","laboratorio",
    "clinic","clínica","unidad","service","servicio",
    # italiano
    "dipartimento","dip.","dipart.","sezione"
]
ORG_KEYWORDS = ORG_PRIMARY + ORG_SECONDARY

# ====== Países (canónicos) + alias ======
COUNTRIES = [   "Algeria", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium","Bosnia and Herzegovina", "Brazil", "Bulgaria", "Canada", "China",
    "Colombia", "Costa Rica", "Czech Republic", "Denmark", "Egypt", "Chile",  "Ethiopia", "Finland", "France", "Germany", "Ghana",
    "Greece", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland","Israel", "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Lebanon",
    "Lithuania", "Malaysia", "Mexico", "Morocco", "Nepal", "Netherlands",    "New Zealand", "Nigeria", "Norway", "Oman", "Pakistan", "Palestine",
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

COUNTRY_ALIASES = {
    "brasil":"Brazil","méxico":"Mexico","españa":"Spain","perú":"Peru",
    "ee.uu.":"United States","u.s.a.":"United States","usa":"United States","us":"United States","u.s.":"United States",
    "uk":"United Kingdom","england":"United Kingdom","scotland":"United Kingdom","wales":"United Kingdom","northern ireland":"United Kingdom",
    "czechia":"Czech Republic","the netherlands":"Netherlands","republic of korea":"South Korea","korea":"South Korea",
}
COUNTRY_CHOICES_LOWER = [c.lower() for c in COUNTRIES]

# Códigos ISO2 en paréntesis → país
ISO2_MAP = {
    "EC": "Ecuador", "IT": "Italy", "ES": "Spain", "FR": "France", "DE": "Germany",
    "PT": "Portugal", "BR": "Brazil", "AR": "Argentina", "MX": "Mexico", "US": "United States",
    "GB": "United Kingdom", "UK": "United Kingdom", "CL": "Chile", "CO": "Colombia",
    "PE": "Peru", "UY": "Uruguay", "PY": "Paraguay"
}

# ====== Regex/utilidades ======
RE_BRACKETS = re.compile(r'\[[^\]]*\]')
RE_MULTISPACE = re.compile(r'\s{2,}')
RE_POSTAL = re.compile(r'\b\d{4,6}\b')
RE_NUMBLOCK = re.compile(r'\b[\d\-]{2,}\b')

# autores/personas
NAME_LIKE = re.compile(
    r'^(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}\s+[A-Z]\.?(?:\s*[A-Z]\.?)?|'
    r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+,\s*[A-Z]\.?(?:\s*[A-Z]\.?)?)$'
)

# detector genérico de DIRECCIONES (no dependemos de lista de ciudades)
ADDRESS_HINT = re.compile(
    r'\b('
    r'av(?:\.|enida)?|ave\.?|avenue|calle|cll|cra|carrera|cdra|rua|via|viale|piazza|p\.za|strada|srt\.?|'
    r'road|rd\.?|street|st\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|highway|hwy\.?|km|km\.|kilometer|kilometro|'
    r'block|bloco|suite|ste\.?|apt\.?|apto|piso|floor|oficina|office|room|'
    r'postal|po box|box|cp|c\.p\.|zip|zipcode|'
    r'no\.|nº|n°|#'
    r')\b', re.IGNORECASE
)

# Abreviaturas (incluye italiano)
ABBR = [
    (r'\bUniv\b\.?', 'University'),
    (r'\bUniversid\b\.?', 'Universidad'),
    (r'\bInst\b\.?', 'Institute'),
    (r'\bInstit\b\.?', 'Instituto'),
    (r'\bDept\b\.?', 'Department'),
    (r'\bFac\b\.?', 'Faculty'),
    (r'\bSch\b\.?', 'School'),
    (r'\bEsc\b\.?', 'Escuela'),
    (r'\bAcad\b\.?', 'Academy'),
    (r'\bHosp?\b\.?', 'Hospital'),
    (r'\bCtr\b\.?|\bCtr\.\b|\bCtrs\b', 'Center'),
    (r'\bMin\b\.?|\bMinist\b\.?', 'Ministry'),
    (r'\bLab\b\.?', 'Laboratory'),
    # italiano
    (r'\bUniv\b\.?\s+degli\s+Studi\b', 'Università degli Studi'),
    (r'\bUniv\b\.?', 'Università'),
    (r'\bIstit\b\.?', 'Istituto'),
    (r'\bDip\b\.?', 'Dipartimento'),
    (r'\bDipart\b\.?', 'Dipartimento'),
    (r'\bPolitec\b\.?', 'Politecnico'),
]

# ---------- helpers ----------
def remove_brackets_with_count(text: str):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    matches = RE_BRACKETS.findall(text)
    cleaned = RE_BRACKETS.sub('', text)
    return cleaned, len(matches)

def has_org_keywords(s: str) -> bool:
    if not s: return False
    t = s.lower()
    return any(k in t for k in ORG_KEYWORDS)

def expand_abbrev_safe(s: str) -> str:
    if not s: return s
    if has_org_keywords(s):  # ya tiene palabras completas
        return s
    for pat, repl in ABBR:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s

def base_clean(s: str) -> str:
    s = s or ""
    s = s.replace('&', ' & ')
    s = RE_POSTAL.sub(' ', s)
    s = RE_NUMBLOCK.sub(' ', s)
    s = RE_MULTISPACE.sub(' ', s)
    return s.strip(' ,;')

def canonical_country(raw_tail: str) -> str|None:
    if not raw_tail: return None
    t = raw_tail.strip()
    # ISO2 entre paréntesis, ej: (EC)
    m = re.fullmatch(r'\((?P<code>[A-Z]{2})\)', t)
    if m:
        return ISO2_MAP.get(m.group('code'))
    tl = t.lower()
    if tl in COUNTRY_ALIASES: return COUNTRY_ALIASES[tl]
    if tl in COUNTRY_CHOICES_LOWER: return COUNTRIES[COUNTRY_CHOICES_LOWER.index(tl)]
    cand, score, _ = process.extractOne(t, COUNTRIES, scorer=fuzz.WRatio)
    return cand if score >= 88 else None

def is_country_segment(seg: str) -> bool:
    return canonical_country(seg) is not None

def is_addressy(seg: str) -> bool:
    s = seg.strip()
    if not s: return True
    if sum(ch.isdigit() for ch in s) >= 2:  # números (cp, calle, etc.)
        return True
    if ADDRESS_HINT.search(s):
        return True
    return False

def normalize_org_text(s: str) -> str:
    return re.sub(r'\s{2,}', ' ', s).strip(' ,;')

def _prefer_primary_side(txt: str) -> str:
    """Si hay ' - ' o ' – ', intenta quedarte con el lado que tiene PRIMARY."""
    parts = re.split(r'\s[–-]\s', txt)
    if len(parts) >= 2:
        left, right = parts[0].strip(), parts[-1].strip()
        if any(k in right.lower() for k in ORG_PRIMARY): return right
        if any(k in left.lower()  for k in ORG_PRIMARY): return left
    return txt

# ---------- extracción país / organizaciones ----------
def extract_country(fragment: str) -> str|None:
    parts = [p.strip() for p in fragment.split(',') if p.strip()]
    if not parts: return None
    # a) último segmento (puede ser (IT))
    c = canonical_country(parts[-1])
    if c: return c
    # b) buscar ISO2 en todo el fragmento
    for code in re.findall(r'\(([A-Z]{2})\)', fragment):
        c2 = ISO2_MAP.get(code)
        if c2: return c2
    # c) nombre de país normal o alias
    low = fragment.lower()
    found = None
    for can in COUNTRIES:
        if re.search(r'(?<!\w)'+re.escape(can.lower())+r'(?!\w)', low):
            found = can
    if found: return found
    for alias, target in COUNTRY_ALIASES.items():
        if re.search(r'(?<!\w)'+re.escape(alias)+r'(?!\w)', low):
            return target
    return None

def extract_orgs(fragment: str) -> list[str]:
    """
    Devuelve lista de organizaciones ANTES del país, sin ciudades/direcciones.
    """
    parts = [p.strip() for p in fragment.split(',') if p.strip()]
    if not parts:
        return []

    # índice del país (desde el final)
    country_idx = None
    for i in range(len(parts)-1, -1, -1):
        if is_country_segment(parts[i]):
            country_idx = i
            break

    head = parts[:country_idx] if country_idx is not None else parts
    if not head:
        return []

    # corta basura/dirección al final
    while head and is_addressy(head[-1]) and not has_org_keywords(head[-1]):
        head.pop()
    if not head:
        return []

    head_lower = [h.lower() for h in head]
    idxs_primary = [i for i,p in enumerate(head_lower) if any(k in p for k in ORG_PRIMARY)]
    orgs = []

    if idxs_primary:
        for i in idxs_primary:
            cand = head[i]
            # opcional: incluir siguiente si ayuda y no es dirección ni país
            if i+1 < len(head) and not is_addressy(head[i+1]) and not is_country_segment(head[i+1]):
                if not has_org_keywords(head[i+1]):
                    cand = f"{cand}, {head[i+1]}"
            cand = _prefer_primary_side(cand)
            orgs.append(normalize_org_text(cand))
        return orgs

    # si no hay PRIMARY, intenta SECONDARY y salta a PRIMARY a la derecha
    idx_secondary = next((i for i,p in enumerate(head_lower) if any(k in p for k in ORG_SECONDARY)), None)
    if idx_secondary is not None:
        for j in range(idx_secondary+1, len(head)):
            if any(k in head_lower[j] for k in ORG_PRIMARY):
                return [normalize_org_text(_prefer_primary_side(head[j]))]
        # si no encuentra PRIMARY, usa el secondary limpio
        return [normalize_org_text(head[idx_secondary])]

    # fallback solo si parece institución
    if has_org_keywords(head[0]):
        return [normalize_org_text(head[0])]
    return []

# ---------- pipeline por celda ----------
def split_blocks(cell: str):
    if not cell: return []
    return [b.strip() for b in re.split(r'[;|]', cell) if b.strip()]

def looks_like_person(s: str) -> bool:
    t = s.strip()
    if len(t.split()) <= 5 and NAME_LIKE.match(t):
        return True
    if re.search(r'\b[A-Z]\.\b', t) and not has_org_keywords(t):
        return True
    return False

def extract_pairs_from_cell(cell: str):
    """Devuelve (pairs, removed) con pairs=[(org,country),...]. Solo institución + país."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return [], 0
    cell = str(cell)

    cell, removed = remove_brackets_with_count(cell)
    if not cell.strip():
        return [], removed

    cell = base_clean(cell)
    blocks = split_blocks(cell)

    pairs = []
    for b in blocks:
        if not b:
            continue
        b = expand_abbrev_safe(b)
        org_list = extract_orgs(b)
        cty = extract_country(b)
        for org in org_list:
            if not org or not cty:
                continue
            if looks_like_person(org):
                continue
            if not has_org_keywords(org):
                continue
            pairs.append((org, cty))
    return pairs, removed

# ---------- dedupe y normalización ----------
def dedupe_pairs(pairs, threshold=83):
    """Dedup por (org~similar, país). Conserva la org más larga."""
    out = []
    for org, cty in pairs:
        placed = False
        for i,(o2,c2) in enumerate(out):
            if c2 == cty:
                if fuzz.token_set_ratio(org, o2) >= threshold:
                    out[i] = (org if len(org)>len(o2) else o2, c2)
                    placed = True
                    break
        if not placed:
            out.append((org, cty))
    return out

def normalize_pairs_string(s: str) -> str:
    """Evita '..., Ecuador, Ecuador' y pares duplicados tras ';'."""
    if not s:
        return s
    pairs = [p.strip() for p in s.split(';') if p.strip()]
    out, seen = [], set()
    for p in pairs:
        chunks = [c.strip() for c in p.split(',') if c.strip()]
        if len(chunks) < 2:
            continue
        # colapsar países duplicados al final
        while len(chunks) >= 2:
            last, prev = chunks[-1], chunks[-2]
            c_last = canonical_country(last)
            c_prev = canonical_country(prev)
            if c_last and c_prev and c_last == c_prev:
                chunks.pop()
            else:
                break
        org = ', '.join(chunks[:-1]).strip()
        ctry = canonical_country(chunks[-1]) or chunks[-1]
        key = (org.lower(), str(ctry).lower())
        if key not in seen:
            out.append(f"{org}, {ctry}")
            seen.add(key)
    return '; '.join(out)

def combine_two_cols(a_str: str, b_str: str):
    a_pairs, a_rm = extract_pairs_from_cell(a_str or "")
    b_pairs, b_rm = extract_pairs_from_cell(b_str or "")
    all_pairs = a_pairs + b_pairs
    deduped = dedupe_pairs(all_pairs, threshold=83)
    combined = '; '.join([f"{o}, {c}" for o, c in deduped])
    combined = normalize_pairs_string(combined)  # solo Organización, País
    removed_total = a_rm + b_rm
    return combined, removed_total

# ========== MAIN ==========
df = pd.read_csv(CSV_IN).fillna("")

if COL_A not in df.columns or COL_B not in df.columns:
    raise ValueError("❌ Faltan columnas esperadas.")

results = df.apply(lambda r: combine_two_cols(r[COL_A], r[COL_B]), axis=1)
df["Combined_affiliations"] = results.map(lambda x: x[0])
df["Removed_bracket_segments"] = results.map(lambda x: x[1])

# ====== REPORTE ======
total_rows = len(df)
empty_rows = int((df["Combined_affiliations"].fillna("").str.strip() == "").sum())
pair_counts = df["Combined_affiliations"].fillna("").str.count(r';') + (df["Combined_affiliations"].str.len() > 0).astype(int)
total_pairs = int(pair_counts.sum())

country_counter = Counter()
for s in df["Combined_affiliations"].fillna(""):
    for p in [x.strip() for x in s.split(';') if x.strip()]:
        country = (p.split(',')[-1] or "").strip()
        if country:
            country_counter[country] += 1
top_countries = country_counter.most_common(10)

print("\n—— REPORTE —————————————————————")
print(f"Filas totales: {total_rows:,}")
print(f"Filas sin resultado (vacías): {empty_rows:,}  ({empty_rows/total_rows:.2%})")
print(f"Pares (Org, País) extraídos: {total_pairs:,}")
print("Top 10 países:")
for c, n in top_countries:
    print(f"  - {c}: {n:,}")

# Guardar
df.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"\n📄 Guardado: {CSV_OUT}")
