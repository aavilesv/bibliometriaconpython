# -*- coding: utf-8 -*-
# Script: affil_remap_postprocess.py
# Uso: ejecuta este archivo DESPUÉS de haber generado la columna "Combined_affiliations"

import re
import unicodedata
import pandas as pd
from rapidfuzz import process, fuzz
from collections import Counter

# ============== CONFIG ==============
CSV_IN  = r"G:/Mi unidad/Artículos cientificos/articulo 1/datawos_scopus_affil_org_country.csv"
CSV_OUT = CSV_IN.replace(".csv", "_remap.csv")

COL_COMBINED = "Combined_affiliations"

# Umbrales
DEDUPE_THRESHOLD    = 90     # para colisiones (fuzzy) de orgs por país
MANUAL_FUZZ_THRESH  = 92     # si activas mapeo manual difuso

# ====== Países (canónicos) + alias ======
COUNTRIES = [ "Algeria","Argentina","Australia","Austria","Bangladesh","Belgium",
    "Bosnia and Herzegovina","Brazil","Bulgaria","Canada","China","Colombia","Costa Rica",
    "Czech Republic","Denmark","Egypt","Chile","Ethiopia","Finland","France","Germany","Ghana",
    "Greece","Hungary","India","Indonesia","Iran","Iraq","Ireland","Israel","Italy","Japan","Jordan",
    "Kazakhstan","Kenya","Lebanon","Lithuania","Malaysia","Mexico","Morocco","Nepal","Netherlands",
    "New Zealand","Nigeria","Norway","Oman","Pakistan","Palestine","Peru","Philippines","Poland",
    "Portugal","Qatar","Romania","Russia","Rwanda","Saudi Arabia","Senegal","Serbia","Singapore",
    "Slovenia","South Africa","South Korea","Spain","Sri Lanka","Sweden","Switzerland","Taiwan",
    "Thailand","Tunisia","Turkey","Ukraine","United Arab Emirates","United Kingdom","United States",
    "Uzbekistan","Vietnam","Zimbabwe","Ivory Coast","Kuwait","Croatia","Afghanistan","Albania",
    "Andorra","Angola","Armenia","Azerbaijan","Bahamas","Bahrain","Barbados","Belarus","Belize",
    "Benin","Bhutan","Bolivia","Botswana","Brunei Darussalam","Burkina Faso","Burundi","Cabo Verde",
    "Cambodia","Cameroon","Central African Republic","Chad","Comoros","Congo","Cuba","Cyprus",
    "Djibouti","Dominican Republic","Ecuador","El Salvador","Equatorial Guinea","Eritrea","Estonia",
    "Eswatini","Fiji","Gabon","Gambia","Georgia","Grenada","Guinea","Guinea-Bissau","Guyana","Haiti",
    "Iceland","Jamaica","Kiribati","Kyrgyzstan","Laos","Latvia","Lesotho","Liberia","Libya",
    "Liechtenstein","Luxembourg","Madagascar","Malawi","Maldives","Mali","Malta","Marshall Islands",
    "Mauritania","Mauritius","Micronesia","Monaco","Mongolia","Montenegro","Mozambique","Myanmar",
    "Namibia","Nauru","Niger","North Macedonia","Palau","Panama","Papua New Guinea","Paraguay",
    "Rwanda","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Samoa",
    "San Marino","Sao Tome and Principe","Seychelles","Sierra Leone","Slovakia","Solomon Islands",
    "Somalia","South Sudan","Sudan","Suriname","Tajikistan","Tanzania","Timor-Leste","Togo","Tonga",
    "Trinidad and Tobago","Turkmenistan","Tuvalu","Uganda","Uruguay","Vanuatu","Venezuela","Yemen",
    "Zambia","Swaziland"
]
COUNTRY_CHOICES_LOWER = [c.lower() for c in COUNTRIES]

def _canonical_country(x: str) -> str|None:
    if not x: return None
    t = x.strip()
    tl = t.lower()
    if tl in COUNTRY_CHOICES_LOWER:
        return COUNTRIES[COUNTRY_CHOICES_LOWER.index(tl)]
    cand = process.extractOne(t, COUNTRIES, scorer=fuzz.WRatio)
    return cand[0] if cand and cand[1] >= 88 else None

# ====== CLAVES para elegir SOLO UNA ORG (por prioridad) ======
KW_C1 = ["university","universidad","universidade","università","université","hochschule",
         "polytechnic","politecnico","politecnica","politechnic"]
KW_C2 = ["institute","instituto","istituto","institution","college","academy","academia","hospital"]
KW_C3 = ["centre","center","centro"]
KW_C4 = ["faculty","facultad","fakultät","school","escuela","école","department","departamento",
         "dipartimento","dip.","dipart.","sezione","laboratory","laboratorio","lab","clinic","clínica",
         "unidad","unit","service","servicio","observatorio","observatory","authority","autoridad",
         "council","consejo"]
PRIORITY_CLASSES = [KW_C1, KW_C2, KW_C3, KW_C4]

def _class_of(seg_low: str) -> int|None:
    for cls_idx, kwlist in enumerate(PRIORITY_CLASSES, start=1):
        if any(k in seg_low for k in kwlist):
            return cls_idx
    return None

# ====== Firma y utilidades para dedupe ======
_STOPWORDS_ORG = {
    "the","of","de","di","da","do","del","la","las","los","y","e","&",
    "university","universidad","universidade","università","université","hochschule",
    "college","polytechnic","politecnico","politecnica","politechnic",
    "institute","instituto","istituto","institution","academy","academia",
    "hospital","centre","center","centro","foundation","fundacion","fundação","fundacao",
    "ministry","ministerio","cnr","conicet","csic","cnrs","planck","helmholtz","museum","museo","museu","musée","musei"
}
def _strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

def _org_signature(name: str) -> str:
    t = _strip_accents(name).lower()
    t = re.sub(r'[^a-z0-9\s&]', ' ', t)
    toks = [w for w in t.split() if w not in _STOPWORDS_ORG and len(w) > 1]
    return ' '.join(sorted(set(toks)))

# ====== Mapeo manual (rellena con tus equivalencias) ======
# clave: forma que quieres reemplazar (inglés/alias) -> valor: forma canónica
MANUAL_ORG_MAP = {
    # "State University of Milagro": "Universidad Estatal de Milagro",
    # "Technical University of Machala": "Universidad Técnica de Machala",
    # "UDLA": "Universidad de Las Américas",
}

# (Opcional) catálogo de nombres destino para reforzar canonicalización difusa al mapear
ORG_CANON_CATALOG = [
    # "Universidad Estatal de Milagro",
    # "Universidad Técnica de Machala",
    # "Universidad de Las Américas",
    # "Universitat de València",
]

def _apply_manual_exact(org: str) -> str:
    if not org: return org
    norm = re.sub(r'\s+', ' ', org).strip()
    for k, v in MANUAL_ORG_MAP.items():
        if norm.lower() == re.sub(r'\s+', ' ', k).strip().lower():
            return v
    return org

def _apply_manual_fuzzy(org: str, threshold: int = MANUAL_FUZZ_THRESH) -> str:
    """Si no hubo match exacto y tienes un catálogo, mapea a la canónica más parecida."""
    if not ORG_CANON_CATALOG:
        return org
    cand = process.extractOne(org, ORG_CANON_CATALOG, scorer=fuzz.WRatio)
    return cand[0] if cand and cand[1] >= threshold else org

# ====== Elegir UNA sola organización si aparece "org1, org2, País" ======
def _pick_primary_from_chunk(org_chunk: str) -> str:
    parts = [p.strip() for p in org_chunk.split(',') if p.strip()]
    cands = []
    for idx, seg in enumerate(parts):
        low = seg.lower()
        cls = _class_of(low)
        if cls is None:
            continue
        # descartamos trozos que parecen unidades/direcciones/personas (muy básico aquí)
        if re.search(r'\b(no\.|nº|n°|km|zip|cp|c\.p\.|street|st\.|road|rd\.|av|avenida|calle|suite|apt|piso)\b', low):
            continue
        if re.match(r'^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}$', seg):
            continue
        cands.append((cls, idx, seg))
    if not cands:
        return parts[0] if parts else ""
    cands.sort(key=lambda x: (x[0], x[1]))  # mejor clase y más a la izquierda
    return cands[0][2]

# ====== Dedupe por país (firma + fuzzy) ======
def _dedupe_pairs(pairs, threshold=DEDUPE_THRESHOLD):
    out, sigs = [], []
    for org, cty in pairs:
        sig = _org_signature(org)
        placed = False
        for i, (o2, c2) in enumerate(out):
            if c2 != cty:
                continue
            if sigs[i] == sig or fuzz.token_set_ratio(org, o2) >= threshold:
                out[i] = (org if len(org) > len(o2) else o2, c2)
                sigs[i] = _org_signature(out[i][0])
                placed = True
                break
        if not placed:
            out.append((org, cty)); sigs.append(sig)
    return out

# ====== Reescritura por fila ======
def remap_row(s: str) -> str:
    """
    Entrada: 'Org A, País; Org B, País; ...'
    1) Aplica mapeo manual (exacto + fuzzy opcional).
    2) Si quedó 'org1, org2, País', elige UNA org principal.
    3) Dedupe por país y fuzzy.
    """
    if not isinstance(s, str) or not s.strip():
        return s

    pairs = []
    for p in [x.strip() for x in s.split(';') if x.strip()]:
        chunks = [c.strip() for c in p.split(',') if c.strip()]
        if len(chunks) < 2:
            continue
        country = _canonical_country(chunks[-1]) or chunks[-1]
        org_chunk = ', '.join(chunks[:-1])

        # 1) elegir UNA org si vienen varias separadas por coma
        org = _pick_primary_from_chunk(org_chunk)

        # 2) mapeo manual exacto + fuzzy opcional (comentado si no quieres fuzzy)
        org = _apply_manual_exact(org)
        org = _apply_manual_fuzzy(org, threshold=MANUAL_FUZZ_THRESH)

        pairs.append((org, country))

    # 3) dedupe por país
    pairs = _dedupe_pairs(pairs, threshold=DEDUPE_THRESHOLD)

    # 4) render final
    return '; '.join([f"{o}, {c}" for (o, c) in pairs])

# ============== MAIN ==============
df = pd.read_csv(CSV_IN).fillna("")
if COL_COMBINED not in df.columns:
    raise ValueError(f"❌ La columna '{COL_COMBINED}' no existe en el CSV.")

df[COL_COMBINED] = df[COL_COMBINED].map(remap_row)

# ====== REPORTE ======
total_rows = len(df)
empty_rows = int((df[COL_COMBINED].fillna("").str.strip() == "").sum())

country_counter = Counter()
for s in df[COL_COMBINED].fillna(""):
    for p in [x.strip() for x in s.split(';') if x.strip()]:
        cc = (p.split(',')[-1] or "").strip()
        if cc:
            country_counter[cc] += 1
top_countries = country_counter.most_common(10)

print("\n—— REPORTE REMAP —————————————————————")
print(f"Filas totales: {total_rows:,}")
print(f"Filas vacías en {COL_COMBINED}: {empty_rows:,}  ({(empty_rows/max(total_rows,1)):.2%})")
print("Top países:")
for c, n in top_countries:
    print(f"  - {c}: {n:,}")

df.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"\n📄 Guardado: {CSV_OUT}")
