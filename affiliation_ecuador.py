# -*- coding: utf-8 -*-
import re
import csv
import pandas as pd
from pathlib import Path
from unidecode import unidecode
from typing import Optional  # <-- para Optional[int]

# ================== CONFIG ==================
IN_CSV  = r"G:/Mi unidad/Artículos cientificos/articulo 1/_affil_org_countryfinalizar3.csv"
OUT_CSV = r"G:/Mi unidad/Artículos cientificos/articulo 1/afiliaciones_detectadas_NOALIASESfinal.csv"
INPUT_COL  = "Combined_universities_final"
OUTPUT_COL = "instituciones_detectadas"
SEPARATOR  = ";"

# Activar (True) un fallback fuzzy MUY ESTRICTO
USE_FUZZY_FALLBACK   = True
FUZZY_WRATIO_STRICT  = 90   # 88–92 recomendado; 90 es buen equilibrio
FUZZY_TOKEN_MIN_PART = 90   # similitud mínima por token distintivo en fallback

# ================== LISTA CANÓNICA (solo Ecuador) ==================
CANONICAL = [
    "ESCUELA POLITÉCNICA NACIONAL",
    "ESCUELA SUPERIOR POLITÉCNICA AGROPECUARIA DE MANABÍ",
    "ESCUELA SUPERIOR POLITÉCNICA DE CHIMBORAZO",
    "ESCUELA SUPERIOR POLITÉCNICA DEL LITORAL",
    "FACULTAD LATINOAMERICANA DE CIENCIAS SOCIALES",
    "INSTITUTO DE ALTOS ESTUDIOS NACIONALES",
    "PONTIFICIA UNIVERSIDAD CATÓLICA DEL ECUADOR",
    "UNIVERSIDAD AGRARIA DEL ECUADOR",
    "UNIVERSIDAD ANDINA SIMÓN BOLÍVAR",
    "UNIVERSIDAD BOLIVARIANA DEL ECUADOR",
    "UNIVERSIDAD CASA GRANDE",
    "UNIVERSIDAD CATÓLICA DE CUENCA",
    "UNIVERSIDAD CATÓLICA DE SANTIAGO DE GUAYAQUIL",
    "UNIVERSIDAD CENTRAL DEL ECUADOR",
    "UNIVERSIDAD DE CUENCA",
    "UNIVERSIDAD DE ESPECIALIDADES TURÍSTICAS",
    "UNIVERSIDAD DE GUAYAQUIL",
    "UNIVERSIDAD DE INVESTIGACIÓN DE TECNOLOGÍA EXPERIMENTAL YACHAY",
    "UNIVERSIDAD DE LAS AMÉRICAS",
    "UNIVERSIDAD DE LAS ARTES",
    "UNIVERSIDAD DE LAS FUERZAS ARMADAS (ESPE)",
    "UNIVERSIDAD DE LOS HEMISFERIOS",
    "UNIVERSIDAD DE OTAVALO",
    "UNIVERSIDAD DEL AZUAY",
    "UNIVERSIDAD DEL PACÍFICO ESCUELA DE NEGOCIOS",
    "UNIVERSIDAD DEL RÍO",
    "UNIVERSIDAD ESTATAL AMAZÓNICA",
    "UNIVERSIDAD ESTATAL DE BOLÍVAR",
    "UNIVERSIDAD ESTATAL DE MILAGRO",
    "UNIVERSIDAD ESTATAL DEL SUR DE MANABÍ",
    "UNIVERSIDAD ESTATAL PENÍNSULA DE SANTA ELENA",
    "UNIVERSIDAD IBEROAMERICANA DEL ECUADOR",
    "UNIVERSIDAD INTERCULTURAL DE LAS NACIONALIDADES Y PUEBLOS INDÍGENAS AMAWTAY WASI",
    "UNIVERSIDAD INTERNACIONAL DEL ECUADOR",
    "UNIVERSIDAD LAICA ELOY ALFARO DE MANABÍ",
    "UNIVERSIDAD LAICA VICENTE ROCAFUERTE DE GUAYAQUIL",
    "UNIVERSIDAD METROPOLITANA",
    "UNIVERSIDAD NACIONAL DE CHIMBORAZO",
    "UNIVERSIDAD NACIONAL DE EDUCACIÓN (UNAE)",
    "UNIVERSIDAD NACIONAL DE LOJA",
    "UNIVERSIDAD PARTICULAR DE ESPECIALIDADES ESPÍRITU SANTO",
    "UNIVERSIDAD INTERNACIONAL SEK",
    "UNIVERSIDAD SAN GREGORIO DE PORTOVIEJO",
    "UNIVERSIDAD POLITÉCNICA ESTATAL DEL CARCHI",
    "UNIVERSIDAD POLITÉCNICA SALESIANA",
    "UNIVERSIDAD REGIONAL AMAZÓNICA IKIAM",
    "UNIVERSIDAD REGIONAL AUTÓNOMA DE LOS ANDES",
    "UNIVERSIDAD SAN FRANCISCO DE QUITO",
    "UNIVERSIDAD TÉCNICA DE AMBATO",
    "UNIVERSIDAD TÉCNICA DE BABAHOYO",
    "UNIVERSIDAD TÉCNICA DE COTOPAXI",
    "UNIVERSIDAD TÉCNICA DE MACHALA",
    "UNIVERSIDAD TÉCNICA DE MANABÍ",
    "UNIVERSIDAD TÉCNICA DEL NORTE",
    "UNIVERSIDAD TÉCNICA ESTATAL DE QUEVEDO",
    "UNIVERSIDAD TÉCNICA LUIS VARGAS TORRES DE ESMERALDAS",
    "UNIVERSIDAD TÉCNICA PARTICULAR DE LOJA",
    "UNIVERSIDAD TECNOLÓGICA ECOTEC",
    "UNIVERSIDAD TECNOLÓGICA EMPRESARIAL DE GUAYAQUIL",
    "UNIVERSIDAD TECNOLÓGICA INDOAMÉRICA",
    "UNIVERSIDAD TECNOLÓGICA ISRAEL",
    "UNIVERSIDAD UTE"
]

# ================== ALIASES CONTROLADOS ==================
ALIASES = {
    "UNIVERSIDAD UTE": [
        "universidad tecnologica equinoccial",
        "universidad tecnológica equinoccial",
        "universidad tecnologica equinoccial (ute)",
        "universidad tecnológica equinoccial (ute)",
        "ute university",
        "ute (universidad tecnologica equinoccial)",
        "ute (universidad tecnológica equinoccial)",
        "universidad tec equinoccial", "univ tec equinoccial",
        "universidad ute", "ute"
    ],
    "UNIVERSIDAD DE LAS FUERZAS ARMADAS (ESPE)": [
        "escuela politécnica del ejército", "escuela politecnica del ejercito",
        "escuela politécnica del ejercito", "escuela politecnica del ejército", "espe"
    ],
    "UNIVERSIDAD CATÓLICA DE SANTIAGO DE GUAYAQUIL": [
        "universidad catolica de guayaquil", "universidad católica de guayaquil", "ucsg"
    ],
    "UNIVERSIDAD PARTICULAR DE ESPECIALIDADES ESPÍRITU SANTO": [
        "UNIVERSIDAD DE ESPECIALIDADES ESPÍRITU SANTO"
    ]


    
}

# ================== NORMALIZACIÓN ==================
def norm(s: str) -> str:
    s = unidecode(str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

GENERIC = {
    "universidad","univ","univ.","de","del","la","el","los","las","y",
    "the","of","and","nacional","tecnica","tecnologica","particular",
    "estatal","regional","superior","politecnica","politecnico","escuela",
    "facultad","catolica","tecnologia","tecnologico","experimental","fuerzas",
    "armadas","instituto",
    "ciencias","agrarias","agraria","ingenieria","ingeniería",
    "department","departamento","departament",
    "faculty","college","mechanical","engineering",
    "quimica","químico","quimico","química"
}

def distinctive_tokens(s_norm: str) -> set[str]:
    return {t for t in s_norm.split() if t not in GENERIC}

# Precomputados
CANON_NORM = {c: norm(c) for c in CANONICAL}
CANON_DIST = {c: distinctive_tokens(CANON_NORM[c]) for c in CANONICAL}

def _build_detect_forms(canonicals: list[str], aliases_map: dict[str, list[str]]):
    detect_forms = {}
    alias_owner = {}
    for canon in canonicals:
        base = {norm(canon)}
        for a in aliases_map.get(canon, []):
            base.add(norm(a))
        detect_forms[canon] = base
        for f in base:
            alias_owner[f] = canon
    return detect_forms, alias_owner

DETECT_FORMS, ALIAS_OWNER = _build_detect_forms(CANONICAL, ALIASES)

# ================== BLOQUEADORES NO-EC ==================
NON_EC_BLOCK_TOKENS = {
    "brazil","brasil","parana","paraná","ponta","grossa","curitiba","paranaense",
    "argentina","peru","colombia","mexico","méxico","queretaro","querétaro","qro",
    "chile","uruguay","paraguay","bolivia","spain","españa","italy","italia",
    "portugal","france","francia","usa","united","states","canada","canadá","taiwan","china",
    "india","saudi","arabia","saudiarabia","kingdom","khalid","beijing","lisbon","lisboa"
}
def fragment_has_foreign_markers(frag_norm: str) -> bool:
    return any(tok in frag_norm for tok in NON_EC_BLOCK_TOKENS)

# ================== REGLAS SENSIBLES POR INSTITUCIÓN ==================
SENSITIVE_CANON = {
    "UNIVERSIDAD UTE": {"equinoccial","quito","ecuador","equinoctial","ute"},
    "UNIVERSIDAD TECNOLÓGICA ISRAEL": {"israel","quito"},
    "UNIVERSIDAD TECNOLÓGICA ECOTEC": {"ecotec","guayaquil","samborondon","samborondón"},
    "UNIVERSIDAD TECNOLÓGICA EMPRESARIAL DE GUAYAQUIL": {"empresarial","uteg","guayaquil"},

    "UNIVERSIDAD DE GUAYAQUIL": {"guayaquil","ug"},
    "UNIVERSIDAD CATÓLICA DE SANTIAGO DE GUAYAQUIL": {"catolica","católica","santiago","guayaquil","ucsg"},
    "UNIVERSIDAD LAICA VICENTE ROCAFUERTE DE GUAYAQUIL": {"rocafuerte","vicente","guayaquil","ulvr"},

    "ESCUELA SUPERIOR POLITÉCNICA DEL LITORAL": {"litoral","espol","guayaquil"},
    "ESCUELA SUPERIOR POLITÉCNICA DE CHIMBORAZO": {"chimborazo","riobamba","espoch"},
    "UNIVERSIDAD POLITÉCNICA SALESIANA": {"salesiana","ups"},

    "UNIVERSIDAD REGIONAL AUTÓNOMA DE LOS ANDES": {"uniandes","ambato","riobamba","quevedo"},
    "UNIVERSIDAD REGIONAL AMAZÓNICA IKIAM": {"ikiam","tena","napo"},

    "UNIVERSIDAD TÉCNICA DE AMBATO": {"ambato","tungurahua"},
    "UNIVERSIDAD TÉCNICA DE MANABÍ": {"manabi","manabí","portoviejo"},
    "UNIVERSIDAD TÉCNICA DE MACHALA": {"machala","el","oro"},
    "UNIVERSIDAD TÉCNICA DE BABAHOYO": {"babahoyo","los","rios","ríos"},
    "UNIVERSIDAD TÉCNICA DEL NORTE": {"ibarra","imbabura","norte"},
    "UNIVERSIDAD TÉCNICA DE COTOPAXI": {"cotopaxi","latacunga"},
    "UNIVERSIDAD TÉCNICA ESTATAL DE QUEVEDO": {"quevedo"},

    "PONTIFICIA UNIVERSIDAD CATÓLICA DEL ECUADOR": {"pontificia","quito","puce"},
}

SENSITIVE_FUZZY_RAISE = {
    "UNIVERSIDAD UTE": 95,
    "UNIVERSIDAD TECNOLÓGICA ISRAEL": 93,
    "UNIVERSIDAD TECNOLÓGICA ECOTEC": 93,
    "UNIVERSIDAD TECNOLÓGICA EMPRESARIAL DE GUAYAQUIL": 93,
    "UNIVERSIDAD DE GUAYAQUIL": 92,
    "UNIVERSIDAD CATÓLICA DE SANTIAGO DE GUAYAQUIL": 92,
    "UNIVERSIDAD LAICA VICENTE ROCAFUERTE DE GUAYAQUIL": 92,
    "UNIVERSIDAD REGIONAL AUTÓNOMA DE LOS ANDES": 92,
    "UNIVERSIDAD REGIONAL AMAZÓNICA IKIAM": 92,
}

def _passes_sensitive_rule(canon: str, frag_norm: str) -> bool:
    req = SENSITIVE_CANON.get(canon)
    if not req:
        return True
    frag_tokens = set(frag_norm.split())
    return bool(req & frag_tokens)

# ================== MATCH ==================
def contains_match(frag_norm: str, canon: str) -> bool:
    for form in DETECT_FORMS[canon]:
        if form and form in frag_norm:
            return True
    return False

MIN_DISTINCT_OVERLAP = 2
def token_overlap_match(frag_norm: str, canon: str) -> bool:
    frag_tokens = set(frag_norm.split())
    overlap = frag_tokens & CANON_DIST[canon]
    return len(overlap) >= MIN_DISTINCT_OVERLAP

try:
    from rapidfuzz import fuzz, process

    def _has_near_distinctive_token(frag_norm: str, canon: str) -> bool:
        frag_tokens = frag_norm.split()
        if not frag_tokens:
            return False
        for dtok in CANON_DIST[canon]:
            best = process.extractOne(dtok, frag_tokens, scorer=fuzz.partial_ratio)
            if best and best[1] >= FUZZY_TOKEN_MIN_PART:
                return True
        return False

    def fuzzy_strict_match(frag_norm: str, canon: str, thr_override: Optional[int] = None) -> bool:
        if not USE_FUZZY_FALLBACK:
            return False
        thr = thr_override if thr_override is not None else FUZZY_WRATIO_STRICT
        for form in DETECT_FORMS[canon]:
            score = fuzz.WRatio(frag_norm, form)
            if score >= thr and _has_near_distinctive_token(frag_norm, canon):
                return True
        return False
except Exception:
    def fuzzy_strict_match(frag_norm: str, canon: str, thr_override: Optional[int] = None) -> bool:
        return False

def detect_in_fragment(fragment: str) -> list[str]:
    out = []
    frag_norm = norm(fragment)
    if not frag_norm:
        return out
    if fragment_has_foreign_markers(frag_norm):
        return out

    # 1) Contiene directo + sensible
    for canon in CANONICAL:
        if not _passes_sensitive_rule(canon, frag_norm):
            continue
        if contains_match(frag_norm, canon):
            out.append(canon)

    # 2) Tokens distintivos + sensible
    if not out:
        for canon in CANONICAL:
            if not _passes_sensitive_rule(canon, frag_norm):
                continue
            if token_overlap_match(frag_norm, canon):
                out.append(canon)

    # 3) Fuzzy estricto + sensible + umbral por institución
    if not out:
        for canon in CANONICAL:
            if not _passes_sensitive_rule(canon, frag_norm):
                continue
            thr = SENSITIVE_FUZZY_RAISE.get(canon, FUZZY_WRATIO_STRICT)
            if fuzzy_strict_match(frag_norm, canon, thr_override=thr):
                out.append(canon)

    # Dedupe preservando orden
    seen = set(); dedup = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        dedup.append(x); seen.add(k)
    return dedup

def detect_in_cell(cell: str) -> str:
    if not str(cell).strip():
        return ""
    frags = [f.strip() for f in str(cell).split(SEPARATOR) if f.strip()]
    detected = []
    for frag in frags:
        detected.extend(detect_in_fragment(frag))
    seen = set(); final = []
    for x in detected:
        k = x.lower()
        if k in seen:
            continue
        final.append(x); seen.add(k)
    return "; ".join(final)

# ================== MAIN ==================
df = pd.read_csv(IN_CSV, dtype=str, low_memory=False).fillna("")
if INPUT_COL not in df.columns:
    raise ValueError(f"❌ No existe la columna '{INPUT_COL}'.")

cols_in = list(df.columns); n_in = len(df)
df[OUTPUT_COL] = df[INPUT_COL].apply(detect_in_cell)

ordered = cols_in.copy()
pos = ordered.index(INPUT_COL) + 1
if OUTPUT_COL in ordered: ordered.remove(OUTPUT_COL)
ordered.insert(pos, OUTPUT_COL)
ordered += [c for c in df.columns if c not in ordered]
df = df.reindex(columns=ordered)

assert len(df) == n_in, "ERROR: cambió el número de filas."
for c in cols_in:
    assert c in df.columns, f"ERROR: falta columna original {c}"

vacias = int((df[OUTPUT_COL].str.strip() == "").sum())
print(f"Filas: {len(df):,} | Sin detecciones: {vacias:,} ({vacias/max(len(df),1):.2%})")

p = Path(OUT_CSV)
df.to_csv(p, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
print(f"📄 Guardado: {p}")
