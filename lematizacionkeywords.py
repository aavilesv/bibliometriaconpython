# -*- coding: utf-8 -*-
"""
Normalización de Author Keywords / Index Keywords (IN-PLACE)
- Limpieza robusta
- Corrección ortográfica CONTEXTUAL (JamSpell -> contextualSpellCheck -> SymSpell)
- Lematización spaCy
- Mapa UK→US / sinónimos
- Filtros para tokens basura
- Protección de acrónimos/tecnicismos
"""

import re, unicodedata, time
from collections import OrderedDict, Counter
from datetime import datetime
from functools import lru_cache
import pandas as pd
from unidecode import unidecode

# ===== Dependencias base =====
# pip install symspellpy spacy unidecode rapidfuzz
# python -m spacy download en_core_web_sm
from symspellpy import SymSpell, Verbosity
import pkg_resources
import spacy
from rapidfuzz.distance import Levenshtein

# ========= CONFIG =========
INPUT_CSV  = r"G:\Mi unidad\2025\master ROSSEMARY CATALINA MONTIEL ARREAGA\nuevo artículo latindex\data\datawos_scopus.csv"
OUTPUT_CSV = r"G:\Mi unidad\2025\master ROSSEMARY CATALINA MONTIEL ARREAGA\nuevo artículo latindex\data\datawos_scopusnormalizado.csv"
KW_COLS = ["Author Keywords", "Index Keywords"]

# Modo/agresividad de corrección contextual
PRESSURE = "high"   # "high" | "medium" | "low"

# (Opcional) JamSpell: modelo LM en inglés# -*- coding: utf-8 -*-
"""
Normalización de Author Keywords / Index Keywords (IN-PLACE)
- Limpieza y normalización robusta sin generar letras sueltas ni números
- Corrección ortográfica con SymSpell (lookup_compound)
- Lematización spaCy
- Mapa UK→US / sinónimos
- Filtros para tokens basura
"""

import re
import unicodedata
from collections import OrderedDict
import pandas as pd
from unidecode import unidecode

# --------- Dependencias externas ---------
# pip install symspellpy spacy unidecode
# python -m spacy download en_core_web_sm
from symspellpy import SymSpell, Verbosity
import pkg_resources
import spacy
import re, unicodedata, time
from datetime import datetime
from collections import Counter
# ========== CONFIG ==========

INPUT_CSV  = r"G:\\Mi unidad\\2025\\master ROSSEMARY CATALINA MONTIEL ARREAGA\\nuevo artículo latindex\\data\\datawos_scopusreemplazar.csv"
OUTPUT_CSV = r"G:\\Mi unidad\\2025\\master ROSSEMARY CATALINA MONTIEL ARREAGA\\nuevo artículo latindex\\data\\datawos_scopusnormalizar.csv"
inicio_fecha_hora = datetime.now()
inicio_tiempo = time.perf_counter()
KW_COLS = ["Author Keywords", "Index Keywords"]  # Ajusta a tu CSV

# Palabras/exps a NO lematizar
EXCEPTION_NOUNS = {
 
    "autism spectrum disorder",
 
}

# UK→US / sinónimos
BRIT_US = {
    "behaviour": "behavior", "behaviours": "behavior",
    "organisation": "organization", "organisations": "organizations",

    # agrega lo que vayas encontrando
}

# Términos técnicos a “proteger” en el diccionario
CUSTOM_TERMS = [
    "ecosystem services", "data"
]

# SymSpell params
MAX_EDIT_DISTANCE = 2
PREFIX_LENGTH     = 7

# ====== spaCy ======
nlp = spacy.load("en_core_web_sm")

# ====== SymSpell ======
sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE,
               prefix_length=PREFIX_LENGTH)

# Diccionario base
freq_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym.load_dictionary(freq_path, 0, 1)

# Opcional, bigramas
# bi_path = pkg_resources.resource_filename(
#     "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
# )
# sym.load_bigram_dictionary(bi_path, 0, 2)

# Proteger términos: subir mucho su frecuencia
for phrase in CUSTOM_TERMS:
    for w in phrase.split():
        sym.create_dictionary_entry(w, 10000)

# ========= REGEX útiles =========
# Detectar siglas estilo U.S.A. / E.U. etc. -> "USA", "EU"
RE_DOTTED_ACRONYM = re.compile(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b')
# Letras sueltas / sólo dígitos
RE_SINGLE_CHAR   = re.compile(r'^[a-z]$')
RE_ONLY_DIGITS   = re.compile(r'^\d+$')


# ========= FUNCIONES =========
def undot_acronyms(text: str) -> str:
    # "U.S." -> "US", "E.U." -> "EU"
    def repl(m):
        return m.group(0).replace('.', '')
    return RE_DOTTED_ACRONYM.sub(repl, text)

def limpieza_basica(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = undot_acronyms(texto)
    texto = unidecode(texto.lower())
    texto = texto.replace('(', ' ').replace(')', ' ')
    # conservar guiones y apóstrofes
    texto = re.sub(r"[^a-z0-9'\-\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def aplicar_mapa(texto: str, mapping: dict) -> str:
    for k, v in mapping.items():
        texto = re.sub(rf"\b{re.escape(k)}\b", v, texto)
    return texto

def correct_with_symspell(term: str) -> str:
    """Corrige a nivel frase, si es 1 palabra usa lookup."""
    term = term.strip()
    if not term:
        return term
    # Si ya es multi-palabra -> usar lookup_compound
    if " " in term:
        res = sym.lookup_compound(term, max_edit_distance=MAX_EDIT_DISTANCE)
        return res[0].term if res else term
    # Si es una sola palabra "corta" no tocar
    if len(term) <= 2:
        return term
    res = sym.lookup(term, Verbosity.CLOSEST, max_edit_distance=MAX_EDIT_DISTANCE)
    return res[0].term if res else term

def filtrar_basura(term: str) -> str:
    """Elimina tokens de 1 letra, sólo dígitos, etc."""
    toks = []
    for t in term.split():
        if RE_SINGLE_CHAR.match(t):
            continue
        if RE_ONLY_DIGITS.match(t):
            continue
        toks.append(t)
    return " ".join(toks).strip()

def lemmatize_spacy(texto: str) -> str:
    doc = nlp(texto)
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        orig = tok.text
        lem  = tok.lemma_
        if tok.tag_ == "VBG":           # gerundio
            out.append(orig)
        elif tok.pos_ == "NOUN" and orig in EXCEPTION_NOUNS:
            out.append(orig)
        else:
            out.append(lem)
    return " ".join(out).strip()

def normalize_single_keyword(kw: str) -> str:
    if not kw:
        return ""
    # 1) Limpieza
    kw = limpieza_basica(kw)
    if not kw:
        return ""
    # 2) Corrección SymSpell
    kw = correct_with_symspell(kw)
    # 3) Filtrar tokens basura (puede quedar algo tras SymSpell)
    kw = filtrar_basura(kw)
    if not kw:
        return ""
    # 4) Lematizar
    kw = lemmatize_spacy(kw)
    # 5) UK→US / sinónimos
    kw = aplicar_mapa(kw, BRIT_US)
    # 6) Filtrar otra vez (por si lematización dejó monos)
    kw = filtrar_basura(kw)
    return kw

def normalize_cell_keywords(cell: str) -> str:
    if not isinstance(cell, str):
        return ""
    raw = [t.strip() for t in cell.split(';') if t.strip()]
    norm_terms = []
    seen = set()
    for term in raw:
        norm = normalize_single_keyword(term)
        if norm and norm not in seen:
            norm_terms.append(norm)
            seen.add(norm)
    return "; ".join(norm_terms)

def contar_unicos(col, df_input):
    all_keywords = (
        df_input[col]
        .dropna()
        .str.split(';')
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
    )
    return all_keywords.nunique()


# ========= MAIN =========
if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV).fillna("")

    print("Antes de normalizar:")
    for c in KW_COLS:
        if c in df.columns:
            print(f"  {c}: {contar_unicos(c, df)} keywords únicas")
        else:
            print(f"  ⚠️ Columna '{c}' no existe en el CSV")

    for col in KW_COLS:
        if col not in df.columns:
            continue
        print(f"\nNormalizando columna: {col} ...")
        df[col] = df[col].apply(normalize_cell_keywords)

    print("\nDespués de normalizar:")
    for c in KW_COLS:
        if c in df.columns:
            print(f"  {c}: {contar_unicos(c, df)} keywords únicas")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print("\n✅ Archivo guardado en:", OUTPUT_CSV)
    
	   # Tiempos
fin_fecha_hora = datetime.now()
fin_tiempo = time.perf_counter()
tiempo_transcurrido = fin_tiempo - inicio_tiempo

print("\n" + "="*60)
print("📅 Inicio:", inicio_fecha_hora.strftime("%Y-%m-%d %H:%M:%S"))
print("🕒 Fin   :", fin_fecha_hora.strftime("%Y-%m-%d %H:%M:%S"))
print(f"⏱️ Tiempo total de ejecución: {tiempo_transcurrido:.2f} s")
print("📁 CSV guardado en:", OUTPUT_CSV)
print("="*60)


JAMSPELL_MODEL_PATH = r"G:\models\jamspell\en.bin"   # cambia si lo tienes en otro sitio

# Palabras/frases a NO lematizar (en minúsculas)
EXCEPTION_NOUNS = {"home literacy environment", "home numeracy environment", "home learning environment",
    "shared reading", "dialogic reading", "guided play", "serve and return",
    "responsive caregiving", "home-based", "home visit", "home visits", "home visiting",
    "house calls", "child-directed speech", "parent-child interaction", "parent–child interaction",
    "parent-child interactions", "parent–child interactions", "parent-child relations",
    "mother-child relations", "mother child relation", "child parent relation",
    "play and playthings", "learning environment", "home environment", "home care",

    # Resultados / constructos cognitivos y de lenguaje
    "executive function", "executive functions", "executive functioning",
    "working memory", "inhibitory control", "cognitive flexibility",
    "emergent literacy", "early literacy", "school readiness",
    "vocabulary development", "phonological awareness",
    "language development", "language ability", "language delay",
    "language development disorders", "developmental language disorder",
    "oral language", "receptive language", "theory of mind",
    "emotion regulation", "self-control", "self control",
    "social cognition", "social communication", "verbal communication",
    "nonverbal communication", "interpersonal communication",
    "social interaction", "social competence", "attention deficit disorder",
    "attention deficit hyperactivity disorder",

    # Poblaciones y rangos etarios (frases)
    "preschool child", "preschool children", "preschool-children",
    "young children", "school child", "infant newborn",
    "early childhood", "early-childhood", "middle aged",
    
    # Diseños y tipos de estudio
    "randomized controlled trial", "randomized controlled trial (topic)",
    "randomized controlled trials as topic", "clinical trial",
    "controlled clinical trial", "longitudinal study", "longitudinal studies",
    "cross-sectional study", "cross-sectional studies", "case-control study",
    "case-control studies", "cohort study", "cohort studies",
    "prospective study", "prospective studies", "comparative study",
    "comparative effectiveness", "observational study", "pilot study",
    "feasibility study", "intervention study", "retrospective study",
    "multicenter study", "systematic review", "meta-analysis", "meta analysis",
    "qualitative research", "qualitative analysis", "exploratory research",
    "descriptive research", "clinical assessment",

    # Instrumentos / escalas / tests
    "bayley scales of infant development",
    "behavior rating inventory of executive function",
    "child behavior checklist",
    "vineland adaptive behavior scale",
    "autism diagnostic observation schedule",
    "wechsler intelligence scale for children",
    "wechsler preschool and primary scale of intelligence",
    "mullen scales of early learning",
    "peabody picture vocabulary test",
    "strengths and difficulties questionnaire",
    "rating scale", "likert scale", "interrater reliability",
    "test retest reliability", "regression analysis", "cluster analysis",
    "psychomotor performance", "speech-language pathology",
    "speech language pathologist", "neuropsychological assessment",
    "structural equation modeling", "electroencephalogram",

    # Condiciones / diagnósticos (frases)
    "autism spectrum disorder", "autism spectrum disorders",
    "hearing impairment", "hearing loss", "intellectual disability",
    "developmental disabilities", "down syndrome", "cochlear implants",
    "cochlear implantation", "cochlea prosthesis", "developmental delay",
    "speech disorder", "mental health", "mental disease",
    "childhood disease", "traumatic brain injury", "prematurity",

    # Salud pública / servicios / educación
    "public health", "primary health care", "primary medical care",
    "health care delivery", "health education", "health status",
    "health survey", "primary school", "day care",
    "early childhood education", "early childhood development",
    "child health care", "child care", "child welfare",

    # Factores socioeconómicos y demográficos
    "socioeconomic status", "socioeconomic-status", "lowest income group",
    "family income", "household income", "low-income",
    "middle income country", "social class", "social status",
    "rural population", "urban population", "developing countries",
    "rural area", "united states", "south africa", "australia",
    "bangladesh", "brazil", "india", "tanzania", "uganda", "hispanic",

    # Otras frases técnicas frecuentes
    "child behavior", "child development", "child language",
    "child health", "motor development", "motor skills",
    "academic achievement", "brain development", "brain function",
    "nuclear magnetic resonance imaging", "magnetic resonance imaging",
    "functional magnetic resonance imaging", "eye tracking",
    "information processing", "physical activity", "social environment",
    "environmental exposure", "environmental factor",
    "child nutrition", "gestational age", "birth weight",
    "parental involvement", "parental behavior", "parental stress",
    "parental attitude", "feeding behavior", "breast feeding",
    "domestic violence", "foster care", "child rearing",
    "cerebral palsy", "human immunodeficiency virus infection",
    "covid-19", "coronavirus disease 2019",

    # Variantes con guion / formas a conservar tal cual
    "working-memory", "language-development", "cognitive-development",
    "self-regulation", "accelerating language-development",
    "child-directed speech", "academic-achievement", "head-start",
    "individual-differences",

    # Acrónimos que no se deben tocar
    "ASD", "asd", "ADHD", "adhd", "HIV"}

# UK→US / sinónimos
BRIT_US = {
    "behaviour": "behavior", "behaviours": "behavior",
    "organisation": "organization", "organisations": "organizations",
}

# Acrónimos/tecnicismos a proteger (minúsculas)
PROTECT_TOKENS = {
    
}

# Regex útil
RE_DOTTED_ACRONYM = re.compile(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b')
RE_SINGLE_CHAR    = re.compile(r'^[a-z]$')
RE_ONLY_DIGITS    = re.compile(r'^\d+$')

# ====== spaCy (para lematizar) ======
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ====== SymSpell (fallback) ======
MAX_EDIT_DISTANCE = 2
PREFIX_LENGTH     = 7
sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE,
               prefix_length=PREFIX_LENGTH)
freq_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym.load_dictionary(freq_path, 0, 1)

# ====== Intentar JamSpell ======
JAMSPELL_READY = False
try:
    from jamspell import TSpellCorrector
    _jam = TSpellCorrector()
    JAMSPELL_READY = _jam.LoadLangModel(JAMSPELL_MODEL_PATH)
except Exception:
    JAMSPELL_READY = False

# ====== Intentar contextualSpellCheck (spaCy transformer) ======
CSC_READY = False
try:
    import contextualSpellCheck
    _nlp_csc = spacy.load("en_core_web_trf")  # requiere spacy-transformers/torch
    contextualSpellCheck.add_to_pipe(_nlp_csc)
    CSC_READY = True
except Exception:
    CSC_READY = False

# ========= UTIL =========
def undot_acronyms(text: str) -> str:
    return RE_DOTTED_ACRONYM.sub(lambda m: m.group(0).replace('.', ''), text)

def limpieza_basica(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = undot_acronyms(texto)
    texto = unidecode(texto.lower())
    texto = texto.replace('(', ' ').replace(')', ' ')
    texto = re.sub(r"[^a-z0-9'\-\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def aplicar_mapa(texto: str, mapping: dict) -> str:
    for k, v in mapping.items():
        texto = re.sub(rf"\b{re.escape(k)}\b", v, texto)
    return texto

def filtrar_basura(term: str) -> str:
    toks = []
    for t in term.split():
        if RE_SINGLE_CHAR.match(t):   # letras sueltas
            continue
        if RE_ONLY_DIGITS.match(t):   # solo dígitos
            continue
        toks.append(t)
    return " ".join(toks).strip()

def lemmatize_spacy(texto: str) -> str:
    doc = nlp(texto)
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        orig = tok.text
        lem  = tok.lemma_
        if tok.tag_ == "VBG":                     # gerundio
            out.append(orig)
        elif tok.pos_ == "NOUN" and orig in EXCEPTION_NOUNS:
            out.append(orig)
        else:
            out.append(lem)
    return " ".join(out).strip()

# ====== Protección de tokens técnicos (mask/unmask) ======
_PROT_MARK = "\uFFF0"  # marcador poco común
def _needs_protect(tok: str) -> bool:
    if tok in PROTECT_TOKENS: return True
    if any(ch.isdigit() for ch in tok): return True
    if re.search(r"[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+", tok): return True
    if "-" in tok and len(tok) <= 5: return True  # ej: sd-wan, l2tp
    return False

def mask_protected(text: str):
    toks = text.split()
    mapping = {}
    out = []
    idx = 0
    for t in toks:
        if _needs_protect(t):
            key = f"{_PROT_MARK}{idx}"
            mapping[key] = t
            out.append(key)
            idx += 1
        else:
            out.append(t)
    return " ".join(out), mapping

def unmask_protected(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

# ====== Aceptación de cambios según "presión" ======
def accept_change(orig: str, new: str) -> bool:
    if orig == new: return True
    d = Levenshtein.distance(orig, new)
    if PRESSURE == "high":
        return d <= 4  # acepta más cambios
    if PRESSURE == "low":
        return d <= 2  # muy conservador
    return d <= 3      # medium

# ====== Corrección contextual (JamSpell -> CSC -> SymSpell) ======
def correct_phrase_contextual(phrase: str) -> str:
    if not phrase.strip(): 
        return phrase

    # 1) proteger tecnicismos
    masked, mapping = mask_protected(phrase)

    corrected = masked

    # 2) JamSpell (si está listo)
    if JAMSPELL_READY:
        try:
            corrected = _jam.FixFragment(masked)
        except Exception:
            pass

    # 3) contextualSpellCheck (si JamSpell no cambió o no disponible)
    if (not JAMSPELL_READY or corrected == masked) and CSC_READY:
        try:
            doc = _nlp_csc(masked)
            if getattr(doc._, "performed_spellCheck", False):
                out = getattr(doc._, "outcome_spellCheck", masked)
                corrected = out
        except Exception:
            pass

    # 4) Fallback: SymSpell a nivel frase
    if corrected == masked:
        res = sym.lookup_compound(masked, max_edit_distance=MAX_EDIT_DISTANCE)
        if res:
            corrected = res[0].term

    # 5) Post-filtro por presión (token a token)
    o_toks = masked.split()
    c_toks = corrected.split()
    if len(o_toks) == len(c_toks):
        c_toks2 = []
        for o, c in zip(o_toks, c_toks):
            c_toks2.append(c if accept_change(o, c) else o)
        corrected = " ".join(c_toks2)

    # 6) restaurar protegidos
    corrected = unmask_protected(corrected, mapping)
    return corrected

# ====== Normalización por término ======
@lru_cache(maxsize=200_000)
def normalize_single_keyword(kw: str) -> str:
    if not kw:
        return ""
    # 1) Limpieza
    kw = limpieza_basica(kw)
    if not kw:
        return ""
    # 2) Corrección contextual (frase completa)
    kw = correct_phrase_contextual(kw)
    # 3) Filtrar basura
    kw = filtrar_basura(kw)
    if not kw:
        return ""
    # 4) Lematizar
    kw = lemmatize_spacy(kw)
    # 5) UK→US / sinónimos
    kw = aplicar_mapa(kw, BRIT_US)
    # 6) Filtrar otra vez (por si lematización dejó monos)
    kw = filtrar_basura(kw)
    return kw

def normalize_cell_keywords(cell: str) -> str:
    if not isinstance(cell, str):
        return ""
    raw = [t.strip() for t in cell.split(';') if t.strip()]
    norm_terms, seen = [], set()
    for term in raw:
        norm = normalize_single_keyword(term)
        if norm and norm not in seen:
            norm_terms.append(norm)
            seen.add(norm)
    return "; ".join(norm_terms)

def contar_unicos(col, df_input):
    all_keywords = (
        df_input[col]
        .dropna()
        .str.split(';')
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
    )
    return all_keywords.nunique()

# ========= MAIN =========
if __name__ == "__main__":
    inicio_fecha_hora = datetime.now()
    inicio_tiempo = time.perf_counter()

    df = pd.read_csv(INPUT_CSV).fillna("")

    print("Antes de normalizar:")
    for c in KW_COLS:
        if c in df.columns:
            print(f"  {c}: {contar_unicos(c, df)} keywords únicas")
        else:
            print(f"  ⚠️ Columna '{c}' no existe en el CSV")

    for col in KW_COLS:
        if col not in df.columns:
            continue
        print(f"\nNormalizando columna: {col} ...")
        df[col] = df[col].apply(normalize_cell_keywords)

    print("\nDespués de normalizar:")
    for c in KW_COLS:
        if c in df.columns:
            print(f"  {c}: {contar_unicos(c, df)} keywords únicas")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print("\n✅ Archivo guardado en:", OUTPUT_CSV)

    # Tiempos
    fin_fecha_hora = datetime.now()
    fin_tiempo = time.perf_counter()
    tiempo_transcurrido = fin_tiempo - inicio_tiempo

    print("\n" + "="*60)
    print("📅 Inicio:", inicio_fecha_hora.strftime("%Y-%m-%d %H:%M:%S"))
    print("🕒 Fin   :", fin_fecha_hora.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"⏱️ Tiempo total de ejecución: {tiempo_transcurrido:.2f} s")
    print("📁 CSV guardado en:", OUTPUT_CSV)
    print("="*60)
