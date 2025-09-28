# -*- coding: utf-8 -*-
import re
import pandas as pd
from rapidfuzz import process, fuzz
from unidecode import unidecode
from pathlib import Path
from collections import Counter

# ================== CONFIG ==================
INPUT_XLSX = r"G:\Mi unidad\Artículos cientificos\articulo 1\data\data_openaccess_.xlsx"
SHEET_NAME = 0  # o el nombre de la hoja, por ej. "Hoja1"
COLUMN_TO_PARSE = "Combined_universities"  # o "Affiliations_final"
OUTPUT_XLSX = str(Path(INPUT_XLSX).with_name("data_openaccess__EC_universidades1.xlsx"))
UNMATCHED_CSV = str(Path(INPUT_XLSX).with_name("no_matcheados_universidades.csv"))

# ============ 1) CANÓNICOS + SIGLAS ============
CANON_Y_SIGLA = {
    "ESCUELA POLITECNICA NACIONAL": "EPN",
    "ESCUELA SUPERIOR POLITECNICA AGROPECUARIA DE MANABI": "ESPAM",
    "ESCUELA SUPERIOR POLITECNICA DE CHIMBORAZO": "ESPOCH",
    "ESCUELA SUPERIOR POLITECNICA DEL LITORAL": "ESPOL",
    "FACULTAD LATINOAMERICANA DE CIENCIAS SOCIALES": "FLACSO",
    "INSTITUTO DE ALTOS ESTUDIOS NACIONALES": "IAEN",
    "PONTIFICIA UNIVERSIDAD CATOLICA DEL ECUADOR": "PUCE",
    "UNIVERSIDAD AGRARIA DEL ECUADOR": "UAE",
    "UNIVERSIDAD ANDINA SIMON BOLIVAR": "UASB",
    "UNIVERSIDAD BOLIVARIANA DEL ECUADOR": "UBE",
    "UNIVERSIDAD CASA GRANDE": "UCG",
    "UNIVERSIDAD CATOLICA DE CUENCA": "UCACUE",
    "UNIVERSIDAD CATOLICA DE SANTIAGO DE GUAYAQUIL": "UCSG",
    "UNIVERSIDAD CENTRAL DEL ECUADOR": "UCE",
    "UNIVERSIDAD DE CUENCA": "UCUENCA",
    "UNIVERSIDAD DE ESPECIALIDADES TURISTICAS": "UCT",
    "UNIVERSIDAD DE GUAYAQUIL": "UG",
    "UNIVERSIDAD DE INVESTIGACION DE TECNOLOGIA EXPERIMENTAL YACHAY": "YACHAY TECH",
    "UNIVERSIDAD DE LAS AMERICAS": "UDLA",
    "UNIVERSIDAD DE LAS ARTES": "UARTES",
    "UNIVERSIDAD DE LAS FUERZAS ARMADAS": "ESPE",
    "UNIVERSIDAD DE LOS HEMISFERIOS": "UDLH",
    "UNIVERSIDAD DE OTAVALO": "UO",
    "UNIVERSIDAD DEL AZUAY": "UAZUAY",
    "UNIVERSIDAD DEL PACIFICO ESCUELA DE NEGOCIOS": "UPACIFICO",
    "UNIVERSIDAD DEL RIO": "UDR",
    "UNIVERSIDAD ESTATAL AMAZONICA": "UEA",
    "UNIVERSIDAD ESTATAL DE BOLIVAR": "UEB",
    "UNIVERSIDAD ESTATAL DE MILAGRO": "UNEMI",
    "UNIVERSIDAD ESTATAL DEL SUR DE MANABI": "UNESUM",
    "UNIVERSIDAD ESTATAL PENINSULA DE SANTA ELENA": "UPSE",
    "UNIVERSIDAD IBEROAMERICANA DEL ECUADOR": "UNIBE",
    "UNIVERSIDAD INTERCULTURAL DE LAS NACIONALIDADES Y PUEBLOS INDIGENAS AMAWTAY WASI": "UIAW",
    "UNIVERSIDAD INTERNACIONAL DEL ECUADOR": "UIDE",
    "UNIVERSIDAD LAICA ELOY ALFARO DE MANABI": "ULEAM",
    "UNIVERSIDAD LAICA VICENTE ROCAFUERTE DE GUAYAQUIL": "ULVR",
    "UNIVERSIDAD METROPOLITANA": "UMET",
    "UNIVERSIDAD NACIONAL DE CHIMBORAZO": "UNACH",
    "UNIVERSIDAD NACIONAL DE EDUCACION UNAE": "UNAE",
    "UNIVERSIDAD NACIONAL DE LOJA": "UNL",
    "UNIVERSIDAD PARTICULAR DE ESPECIALIDADES ESPIRITU SANTO": "UEES",
    "UNIVERSIDAD PARTICULAR INTERNACIONAL SEK": "UISEK",
    "UNIVERSIDAD PARTICULAR SAN GREGORIO DE PORTOVIEJO": "USGP",
    "UNIVERSIDAD POLITECNICA ESTATAL DEL CARCHI": "UPEC",
    "UNIVERSIDAD POLITECNICA SALESIANA": "UPS",
    "UNIVERSIDAD REGIONAL AMAZONICA IKIAM": "IKIAM",
    "UNIVERSIDAD REGIONAL AUTONOMA DE LOS ANDES": "UNIANDES",
    "UNIVERSIDAD SAN FRANCISCO DE QUITO": "USFQ",
    "UNIVERSIDAD TECNICA DE AMBATO": "UTA",
    "UNIVERSIDAD TECNICA DE BABAHOYO": "UTB",
    "UNIVERSIDAD TECNICA DE COTOPAXI": "UTC",
    "UNIVERSIDAD TECNICA DE MACHALA": "UTMACH",
    "UNIVERSIDAD TECNICA DE MANABI": "UTM",
    "UNIVERSIDAD TECNICA DEL NORTE": "UTN",
    "UNIVERSIDAD TECNICA ESTATAL DE QUEVEDO": "UTEQ",
    "UNIVERSIDAD TECNICA LUIS VARGAS TORRES DE ESMERALDAS": "UTLVTE",
    "UNIVERSIDAD TECNICA PARTICULAR DE LOJA": "UTPL",
    "UNIVERSIDAD TECNOLOGICA ECOTEC": "ECOTEC",
    "UNIVERSIDAD TECNOLOGICA EMPRESARIAL DE GUAYAQUIL": "UTEG",
    "UNIVERSIDAD TECNOLOGICA INDOAMERICA": "UTI",
    "UNIVERSIDAD TECNOLOGICA ISRAEL": "UISRAEL",
    "UNIVERSIDAD UTE": "UTE"
}
CANON_LIST = list(CANON_Y_SIGLA.keys())

# ============ 2) ALIASES / VARIANTES ============
ALIASES = {
    "UNIVERSIDAD SAN FRANCISCO DE QUITO": [
        "USFQ", "UNIV SAN FRANCISCO DE QUITO", "SAN FRANCISCO DE QUITO UNIVERSITY",
        "UNIVERSITY SAN FRANCISCO DE QUITO", "SAN FRANCISCO UNIVERSITY OF QUITO",
        "SAN FCO DE QUITO UNIV"
    ],
    "ESCUELA SUPERIOR POLITECNICA DEL LITORAL": [
        "ESPOL", "ESCUELA SUP POLITECNICA DEL LITORAL",
        "ESCUELA SUPERIOR POLITECNICA DEL LITORAL (ESPOL)",
        "POLYTECHNIC SCHOOL OF THE LITORAL"
    ],
    "UNIVERSIDAD DE LAS FUERZAS ARMADAS": [
        "ESPE", "UNIV DE LAS FUERZAS ARMADAS", "ARMED FORCES UNIVERSITY"
    ],
    "UNIVERSIDAD UTE": [
        "UTE", "UNIVERSIDAD TECNOLOGICA EQUINOCCIAL", "TECHNOLOGICAL UNIVERSITY EQUINOCCIAL"
    ],
    "PONTIFICIA UNIVERSIDAD CATOLICA DEL ECUADOR": [
        "PUCE", "PONTIFICAL CATHOLIC UNIVERSITY OF ECUADOR"
    ],
    "UNIVERSIDAD TECNICA PARTICULAR DE LOJA": [
        "UTPL", "TECHNICAL PRIVATE UNIVERSITY OF LOJA"
    ],
    "UNIVERSIDAD TECNICA DE MANABI": [
        "UTM", "TECHNICAL UNIVERSITY OF MANABI", "UNIV TEC DE MANABI"
    ],
    "UNIVERSIDAD TECNICA DEL NORTE": [
        "UTN", "TECHNICAL UNIVERSITY OF THE NORTH"
    ],
    "UNIVERSIDAD TECNICA DE MACHALA": [
        "UTMACH", "TECHNICAL UNIVERSITY OF MACHALA"
    ],
    "UNIVERSIDAD INTERNACIONAL DEL ECUADOR": [
        "UIDE", "INTERNATIONAL UNIVERSITY OF ECUADOR"
    ],
    "UNIVERSIDAD POLITECNICA SALESIANA": [
        "UPS", "SALESIAN POLYTECHNIC UNIVERSITY"
    ],
    "ESCUELA POLITECNICA NACIONAL": [
        "EPN", "NATIONAL POLYTECHNIC SCHOOL", "POLYTECHNIC SCHOOL OF QUITO"
    ],
    "UNIVERSIDAD CENTRAL DEL ECUADOR": [
        "UCE", "CENTRAL UNIVERSITY OF ECUADOR"
    ],
    "UNIVERSIDAD TECNOLOGICA EMPRESARIAL DE GUAYAQUIL": [
        "UTEG", "TECHNOLOGICAL BUSINESS UNIVERSITY OF GUAYAQUIL"
    ],
    "UNIVERSIDAD DE LAS AMERICAS": [
        "UDLA", "UNIVERSITY OF THE AMERICAS QUITO", "UNIV DE LAS AMERICAS"
    ],
    "UNIVERSIDAD ESTATAL PENINSULA DE SANTA ELENA": [
        "UPSE", "STATE UNIVERSITY PENINSULA OF SANTA ELENA"
    ],
    "UNIVERSIDAD LAICA ELOY ALFARO DE MANABI": [
        "ULEAM", "LAICA ELOY ALFARO MANABI UNIVERSITY"
    ],
    "UNIVERSIDAD CATOLICA DE SANTIAGO DE GUAYAQUIL": [
        "UCSG", "CATHOLIC UNIVERSITY OF SANTIAGO DE GUAYAQUIL"
    ],
    "UNIVERSIDAD TECNICA ESTATAL DE QUEVEDO": [
        "UTEQ", "STATE TECHNICAL UNIVERSITY OF QUEVEDO"
    ],
    "UNIVERSIDAD TECNICA DE AMBATO": [
        "UTA", "TECHNICAL UNIVERSITY OF AMBATO"
    ],
    "UNIVERSIDAD NACIONAL DE LOJA": [
        "UNL", "NATIONAL UNIVERSITY OF LOJA"
    ],
    "UNIVERSIDAD DE CUENCA": [
        "UCUENCA", "UNIVERSITY OF CUENCA"
    ],
    "UNIVERSIDAD DEL AZUAY": [
        "UAZUAY", "UNIVERSITY OF AZUAY"
    ],
    "UNIVERSIDAD CATOLICA DE CUENCA": [
        "UCACUE", "CATHOLIC UNIVERSITY OF CUENCA"
    ],
    "UNIVERSIDAD ESTATAL DE MILAGRO": [
        "UNEMI", "STATE UNIVERSITY OF MILAGRO"
    ],
    "UNIVERSIDAD TECNICA LUIS VARGAS TORRES DE ESMERALDAS": [
        "UTLVTE", "TECHNICAL UNIVERSITY LUIS VARGAS TORRES"
    ],
}

# ============ 3) Normalización y utilidades ============
def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unidecode(str(s)).upper()
    s = s.replace("&", " Y ").replace("/", " ")
    s = re.sub(r"[^A-Z0-9 \-\.\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canon_display_name(canon: str) -> str:
    sigla = CANON_Y_SIGLA.get(canon)
    return f"{canon} ({sigla})" if sigla else canon

# índice variantes normalizadas -> canónico
variant_to_canon = {}
for canon in CANON_LIST:
    variant_to_canon[normalize_text(canon)] = canon
for canon, variants in ALIASES.items():
    for v in variants:
        variant_to_canon[normalize_text(v)] = canon
TARGETS = list(variant_to_canon.keys())

# ---- Recorte de prefijos "Faculty/School ... University ..." ----
DEPT_PREFIXES = [
    r"INDUSTRY AND CONSTRUCTION",
    r"HIGHER TECHNICAL SCHOOL",
    r"SCHOOL OF", r"FACULTY OF", r"DEPARTMENT OF", r"INSTITUTE OF",
    r"COLLEGE OF", r"ESCUELA", r"FACULTAD", r"DEPARTAMENTO", r"INSTITUTO"
]
DEPT_REGEX = re.compile(rf"^({'|'.join(DEPT_PREFIXES)})\b.*?\b(UNIVERSIDAD|UNIVERSITY|UNIV\.?)\b\s*", re.IGNORECASE)

def strip_dept_prefix(raw: str) -> str:
    if not raw:
        return raw
    return DEPT_REGEX.sub(r"\2 ", raw).strip()

# ---- Tokens y “anclas” ----
STOP_TOKENS = {
    "UNIVERSIDAD","UNIV","UNIVERSITY","POLITECNICA","POLYTECHNIC","TECNICA","TECHNICAL",
    "DE","DEL","LA","EL","LOS","LAS","Y","EN","NACIONAL","ESTATAL","PARTICULAR","REGIONAL",
    "SUPERIOR","ESCUELA","SCHOOL","FACULTY","DEPARTMENT","INSTITUTE","COLLEGE","PONTIFICIA",
    "CATOLICA","CATHOLIC","ARMADAS","FUERZAS","AMERICAS","LATINOAMERICANA","INTERNACIONAL","INTERNATIONAL"
}
CITY_TOKENS = {
    "QUITO","GUAYAQUIL","CUENCA","LOJA","PORTOVIEJO","MANTA","AMBATO","IBARRA",
    "RIOBAMBA","LATACUNGA","ESMERALDAS","MACHALA","TENA","PUYO","MILAGRO","QUEVEDO",
    "SANGOLQUI","OTAVALO","JIPIJAPA","CALCETA","TULCAN"
}
COUNTRY_TOKENS = {"ECUADOR","EC"}

def toks(s: str) -> list:
    return re.findall(r"[A-Z0-9]+", s or "")

def anchor_tokens_from_text(norm: str) -> set:
    t = set(toks(norm))
    return t - STOP_TOKENS - CITY_TOKENS - COUNTRY_TOKENS

def anchor_tokens_from_canon(canon: str) -> set:
    return anchor_tokens_from_text(normalize_text(canon))

# ---- Siglas únicas → canónico ----
ACRONYM_TO_CANON = {sigla: canon for canon, sigla in CANON_Y_SIGLA.items() if sigla}
ACRONYM_KEYS = set(ACRONYM_TO_CANON.keys())

# ---- Similitud compuesta (cubre inicio/medio/final) ----
def composite_similarity(a_norm: str, b_norm: str) -> float:
    s1 = fuzz.WRatio(a_norm, b_norm)
    s2 = fuzz.token_set_ratio(a_norm, b_norm)
    s3 = fuzz.partial_ratio(a_norm, b_norm)
    return 0.5*s1 + 0.3*s2 + 0.2*s3

# ---- Coincidencia de anclas (tolerante a typos) ----
def anchor_overlap_score(text_anchors: set, canon_anchors: set) -> float:
    if not canon_anchors:
        return 0.0
    matched = 0
    for ca in canon_anchors:
        if ca in text_anchors:
            matched += 1
        else:
            # tolera typos leves en tokens ancla
            if text_anchors:
                best = max((fuzz.ratio(ca, t) for t in text_anchors), default=0)
                if best >= 88:  # p.ej., ROCAFUENTE ~ ROCAFUERTE
                    matched += 1
    return matched / max(1, len(canon_anchors))

def dynamic_threshold(base: float, anchor_cov: float) -> float:
    thr = base
    if anchor_cov >= 0.66:
        thr -= 6.0
    elif anchor_cov >= 0.5:
        thr -= 3.0
    return max(82.0, thr)

# ============ 4) Matching robusto ============
def match_to_ec_canon(raw_name: str, base_high=92, base_mid=86):
    if pd.isna(raw_name) or not str(raw_name).strip():
        return None

    raw_clean = strip_dept_prefix(str(raw_name))
    norm = normalize_text(raw_clean)

    # exacto por variante
    if norm in variant_to_canon:
        return variant_to_canon[norm]

    # sigla como token aislado
    tk = set(toks(norm))
    acr_hits = list(tk & ACRONYM_KEYS)
    if len(acr_hits) == 1:
        return ACRONYM_TO_CANON[acr_hits[0]]

    # candidato inicial por variantes (rápido)
    cand_key, _, _ = process.extractOne(norm, TARGETS, scorer=fuzz.WRatio)
    _ = variant_to_canon[cand_key]  # no lo usamos directamente; pasamos a evaluar todos

    # evalúa todos los canónicos
    best_canon = None
    best_score = -1.0
    best_anchor_cov = 0.0
    text_anchors = anchor_tokens_from_text(norm)

    for canon in CANON_LIST:
        canon_norm = normalize_text(canon)
        score = composite_similarity(norm, canon_norm)
        canon_anchors = anchor_tokens_from_canon(canon)
        cov = anchor_overlap_score(text_anchors, canon_anchors)
        if cov == 0.0:
            score -= 8.0  # sin anclas reales, penaliza fuerte
        else:
            score += min(4.0 * cov, 4.0)  # pequeño premio
        if score > best_score:
            best_score = score
            best_canon = canon
            best_anchor_cov = cov

    # decisión final con umbral dinámico
    thr_high = dynamic_threshold(base_high, best_anchor_cov)
    thr_mid  = dynamic_threshold(base_mid,  best_anchor_cov)

    if best_score >= thr_high:
        return best_canon
    if best_score >= thr_mid and best_anchor_cov >= 0.33:
        return best_canon

    return None

# ============ 5) Extracción en una celda ============
def extract_ec_universities(affil_field: str):
    if pd.isna(affil_field) or not str(affil_field).strip():
        return None
    parts = [p.strip() for p in str(affil_field).split(";") if p.strip()]
    found = []
    for p in parts:
        m = match_to_ec_canon(p)
        if m:
            found.append(m)
    if not found:
        return None
    found = sorted(set(found))
    return "; ".join(canon_display_name(c) for c in found)

# ============ 6) Ejecutar sobre tu Excel ============
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

if COLUMN_TO_PARSE not in df.columns:
    raise ValueError(
        f"La columna '{COLUMN_TO_PARSE}' no existe en el archivo. "
        f"Columnas disponibles: {list(df.columns)}"
    )

# Nueva columna con universidades ecuatorianas normalizadas (o NaN si no hay)
df["Universidades_EC_norm"] = df[COLUMN_TO_PARSE].apply(extract_ec_universities)

# ---- Log de no-matcheados para enriquecer ALIASES ----
def collect_unmatched(row):
    raw = row[COLUMN_TO_PARSE]
    if pd.isna(raw) or not str(raw).strip():
        return []
    no_match = []
    for p in [x.strip() for x in str(raw).split(";") if x.strip()]:
        m = match_to_ec_canon(p)
        if not m:
            no_match.append(p)
    return no_match

unmatched_series = df.apply(collect_unmatched, axis=1)
flat = []
for lst in unmatched_series:
    flat.extend(lst)
cnt = Counter(flat)
if cnt:
    pd.DataFrame(cnt.most_common(), columns=["afiliacion_no_matcheada", "frecuencia"])\
      .to_csv(UNMATCHED_CSV, index=False)

# Guardar resultado
df.to_excel(OUTPUT_XLSX, index=False)
print(f"Listo. Guardado en: {OUTPUT_XLSX}")
if cnt:
    print(f"También se guardó un log de no matcheados en: {UNMATCHED_CSV}")
