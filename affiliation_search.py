# -*- coding: utf-8 -*-
import re
import pandas as pd
from rapidfuzz import process, fuzz

# ========= RUTAS =========
CSV_IN   = r"G:/Mi unidad/Artículos cientificos/articulo 1/datawos_scopus_affil_org_country.csv"
XLSX_BASE = CSV_IN.replace(".csv", "")
XLSX_OUT_GOOD  = XLSX_BASE + "_terms_org_country.xlsx"
XLSX_OUT_OTHER = XLSX_BASE + "_terms_other.xlsx"

COL_COMBINED = "Combined_affiliations"

# Opciones
LOWERCASE = True           # poner todo en minúsculas
DROP_EMPTY = True          # quitar vacíos tras limpiar

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
ISO2_MAP = {
    "EC":"Ecuador","IT":"Italy","ES":"Spain","FR":"France","DE":"Germany","PT":"Portugal",
    "BR":"Brazil","AR":"Argentina","MX":"Mexico","US":"United States","GB":"United Kingdom",
    "UK":"United Kingdom","CL":"Chile","CO":"Colombia","PE":"Peru","UY":"Uruguay","PY":"Paraguay"
}

# ====== Prefijos/fragmentos que indican ORGANIZACIÓN ======
ORG_PREFIXES = [
  'univ','universid','università','universidade',
  'inst','instit','istit','institution',
  'coll','college',
  'scho','school','escu','escuela',
  'acad','academy','academia',
  'fac','facu','facultad',
  'dept','depart','department','departamento',
  'center','centre','centro',
  'minist','ministry','ministerio',
  'hospital','hosp',
  'politec','politecnico','politecnica',
  'cnr','conicet','csic','cnrs','max planck','helmholtz',
   "facu","facultad","fakultät",
   "school","escu","école",
   "depart","depart","dipartimento","dip.","dipart.","sezione",
   "laboratory","laboratorio","lab",
   "clinic","clínica","clinique",
   "unidad","unit","service","servicio",
   "observatorio","observatory",
   "authority","autoridad","council","consejo","museo", "museum"
]

# ---------- helpers ----------
def normalize_piece(s: str) -> str:
    """Limpia espacios, elimina puntuación colgante y colapsa espacios múltiples."""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip(" \t\r\n,;.")
    s = re.sub(r"\s{2,}", " ", s)
    if LOWERCASE:
        s = s.lower()
    return s

def canonical_country(s: str) -> str|None:
    if not s: return None
    t = s.strip()
    m = re.fullmatch(r'\(([A-Z]{2})\)', t)
    if m:
        return ISO2_MAP.get(m.group(1))
    tl = t.lower()
    if tl in COUNTRY_ALIASES: return COUNTRY_ALIASES[tl]
    if tl in COUNTRY_CHOICES_LOWER:
        return COUNTRIES[COUNTRY_CHOICES_LOWER.index(tl)]
    cand = process.extractOne(t, COUNTRIES, scorer=fuzz.WRatio)
    return cand[0] if cand and cand[1] >= 88 else None

def is_country_term(term: str) -> bool:
    return canonical_country(term) is not None

def looks_like_org(term: str) -> bool:
    t = term.lower()
    return any(p in t for p in ORG_PREFIXES)

def explode_terms(series: pd.Series):
    """Explota Combined_affiliations a términos sueltos (separando por ; y ,)."""
    terms = []
    for cell in series.fillna(""):
        if not cell:
            continue
        for chunk in cell.split(';'):
            chunk = chunk.strip()
            if not chunk:
                continue
            for piece in chunk.split(','):
                term = normalize_piece(piece)
                if DROP_EMPTY and not term:
                    continue
                terms.append(term)
    # eliminar duplicados preservando orden
    seen, uniq = set(), []
    for t in terms:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def split_good_vs_other(terms):
    """Separa en (org/país) y otros, preservando orden y sin duplicados."""
    good, other = [], []
    seen_good, seen_other = set(), set()
    for t in terms:
        if is_country_term(t) or looks_like_org(t):
            if t not in seen_good:
                good.append(t); seen_good.add(t)
        else:
            if t not in seen_other:
                other.append(t); seen_other.add(t)
    return good, other

def main():
    df = pd.read_csv(CSV_IN, dtype=str).fillna("")
    if COL_COMBINED not in df.columns:
        raise ValueError(f"❌ No se encontró la columna '{COL_COMBINED}' en {CSV_IN}")

    terms = explode_terms(df[COL_COMBINED])
    good, other = split_good_vs_other(terms)

    # Guardar a Excel (una columna 'term' en cada archivo)
    with pd.ExcelWriter(XLSX_OUT_GOOD, engine="xlsxwriter") as xlw:
        pd.Series(good, name="term").to_excel(xlw, index=False, sheet_name="terms")
    with pd.ExcelWriter(XLSX_OUT_OTHER, engine="xlsxwriter") as xlw:
        pd.Series(other, name="term").to_excel(xlw, index=False, sheet_name="terms")

    print("✅ Exportados:")
    print(f"  • País/Organización: {XLSX_OUT_GOOD}")
    print(f"  • Otros (ciudades/direcciones/etc.): {XLSX_OUT_OTHER}")

if __name__ == "__main__":
    main()
