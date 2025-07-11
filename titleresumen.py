


import sys, re, unidecode, pandas as pd, spacy
from langdetect import detect, LangDetectException
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP

# ------------ 1 ▸ Rutas por defecto ------------
DEFAULT_IN  =r"G:\\Mi unidad\\2025\\codigos bibliometria NPL\\clasificacion_title_abstract.xlsx"
DEFAULT_OUT = r"G:\\Mi unidad\\2025\\codigos bibliometria NPL\\clasificacion_title_abstract_salida2.xlsx"
IN_FILE  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

# ------------ 2 ▸ Carga modelos spaCy ------------
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_md")

# ------------ 3 ▸ Stop-words unificadas ------------
EXTRA_SW = {"background","objective","objectives","methods",
            "results","conclusion","conclusions","study","studies",
            "analysis","introduction","paper","article"}
STOP = EN_STOP | ES_STOP | EXTRA_SW

# ──────────────────────────────────────────────
# 4 ▸ frases clave a “congelar” (puedes añadir más)
#    * El comodín * SOLO al final -> prefijo variable
# ──────────────────────────────────────────────
PHRASES_RAW = [
    # creatividad
    "divergent thinking", "lateral thinking", "creative ideation",
    "idea generation", "ideational fluency", "creative fluency",
    "creative potential", "creative cognition",
    # educación superior
    "higher education", "universit*", "tertiary education", "postsecondary",
    "post-secondary", "college student*", "undergraduate student*",
    "graduate student*", "university student*", "HEI",
    "teacher training college*"
]

REGEX_PATTERNS = []
for ph in PHRASES_RAW:
    ph_lc = ph.lower()
    if ph_lc.endswith("*"):                       # wildcard de prefijo
        prefix = re.escape(ph_lc[:-1].strip())
        pattern = re.compile(rf"\b{prefix}\w*\b", flags=re.I)
        # preserva posibles plurales/sufijos y mantiene "_" en lugar de espacio
        replacement = lambda m, p=prefix: m.group(0).replace(" ", "_")
    else:                                         # coincidencia exacta
        pattern = re.compile(re.escape(ph_lc), flags=re.I)
        replacement = ph_lc.replace(" ", "_")
    REGEX_PATTERNS.append((pattern, replacement))

def freeze_phrases(text: str) -> str:
    """Sustituye frases por versión con '_' para que no se dividan."""
    if not isinstance(text, str): return ""
    txt = text.lower()
    for pat, repl in REGEX_PATTERNS:
        txt = pat.sub(repl, txt)
    return txt

# ──────────────────────────────────────────────
# 5 ▸ utilidades de idioma y limpieza
# ──────────────────────────────────────────────
def lang_of(text: str) -> str:
    try:
        return detect(text[:400])
    except LangDetectException:
        return "es" if any(c in "áéíóúñ" for c in text) else "en"

def lematizar_title_abstract(texto: str) -> list[str]:
    """Lematiza Title+Abstract después de congelar las frases clave."""
    if not isinstance(texto, str) or not texto.strip():
        return []
    texto = unidecode.unidecode(freeze_phrases(texto.lower()))
    nlp = nlp_es if lang_of(texto).startswith("es") else nlp_en
    doc = nlp(texto)
    return [
        tok.lemma_
        for tok in doc
        if tok.is_alpha and len(tok) > 2 and tok.lemma_ not in STOP
    ]

def keywords_a_lista(celda: str) -> list[str]:
    """'kw1; kw2' -> ['kw1', 'kw2'] (minúsculas, sin acentos)."""
    if not isinstance(celda, str):
        return []
    kws = [kw.strip() for kw in celda.split(";") if kw.strip()]
    kws = [unidecode.unidecode(kw.lower()) for kw in kws]
    return kws

# ──────────────────────────────────────────────
# 6 ▸ pipeline
# ──────────────────────────────────────────────
print("📂 Leyendo", IN_FILE)
df = pd.read_excel(IN_FILE, engine="openpyxl")

df["TextoFull"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")

def fila_a_tokens(row) -> str:
    lemas     = lematizar_title_abstract(row["TextoFull"])
    kw_author = keywords_a_lista(row.get("Author Keywords"))
    kw_index  = keywords_a_lista(row.get("Index Keywords"))
    tokens = lemas + kw_author + kw_index          # concatena
    tokens = list(dict.fromkeys(tokens))           # deduplicación
    return ";".join(tokens)

print("⚙️  Procesando filas…")
df["ResumenTokens"] = df.apply(fila_a_tokens, axis=1)

print("💾 Guardando", OUT_FILE)
df.to_excel(OUT_FILE, index=False, engine="openpyxl")
print("✅ Listo: archivo creado sin errores.")