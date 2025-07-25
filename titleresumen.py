'''
pip install pandas openpyxl spacy unidecode langdetect gensim
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_md
Línea	¿Qué instala / descarga?	Para qué se usa en el script
pandas	Manipulación de DataFrames y lectura/escritura Excel.	Leer el archivo fuente y guardar el resultado.
openpyxl	Motor que Pandas utiliza para .xlsx.	Necesario para pd.read_excel(..., engine="openpyxl").
spacy	NLP rápido en CPU.	Lematización y filtrado POS.
unidecode	Transforma caracteres acentuados a ASCII.	Homogeneizar tildes/acentos durante limpieza.
langdetect	Detección rápida de idioma.	Elegir modelo spaCy EN vs ES por fila.
gensim	Extracción estadística de n-gramas.	Aprender bigramas frecuentes.
en_core_web_sm, es_core_news_md	Modelos spaCy inglés y español.	Tokenización + lematización multi-idioma.

'''
import sys, re, unidecode, pandas as pd, spacy
from langdetect import detect, LangDetectException
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from rapidfuzz import fuzz, process


from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
# ─────────────────────────────────────────────
# 1 ▸ Rutas por defecto
# ─────────────────────────────────────────────

DEFAULT_IN  = r"G:\\Mi unidad\\2024\\Msc. Alberto León Batallas\\nuevoarticulo\\artículo final\\data\\wos_scopus.csv"
 
DEFAULT_OUT = r"G:\\Mi unidad\\2024\\Msc. Alberto León Batallas\\nuevoarticulo\\artículo final\\data\\datawos_scopustitleresumen.csv"
IN_FILE  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

# ───────────── 2 · spaCy
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_md")

# ───────────── 3 · Stop-words
EXTRA_SW = {
    "background","objective","objectives","methods","results",
    "conclusion","conclusions","study","studies","analysis",
    "introduction","paper","article"
}
STOP = EN_STOP | ES_STOP | EXTRA_SW

# ───────────── 4 · Frases manuales base
PHRASES_RAW = [
    "divergent thinking","lateral thinking","creative ideation",
    "idea generation","ideational fluency","creative fluency",
    "creative potential","creative cognition",
    "higher education","universit*","tertiary education","postsecondary",
    "post-secondary","college student*","undergraduate student*",
    "graduate student*","university student*","hei",
    "teacher training college*"
]

# ───────────── 5 · Leer datos
print("📂 Leyendo", IN_FILE)
#df = pd.read_excel(IN_FILE, engine="openpyxl")
df = pd.read_csv(IN_FILE)
df["TextoFull"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")

# ───────────── 6 · BIGRAMAS (solo 2-gramas)
print("🔗 Aprendiendo bigramas…")
sentences = [simple_preprocess(text.lower()) for text in df["TextoFull"].fillna("")]
bigram = Phrases(sentences, min_count=5, threshold=15,
                 connector_words=ENGLISH_CONNECTOR_WORDS)
bigram_mod = Phraser(bigram)

def aplicar_bigrams(text):
    return " ".join(bigram_mod[simple_preprocess(text.lower())])

df["TextoFull"] = df["TextoFull"].fillna("").apply(aplicar_bigrams)

# ───────────── 6-bis · Añadir keywords → freeze dinámico
kw_phrases = set()
for col in ["Author Keywords", "Index Keywords"]:
    if col in df.columns:
        kw_phrases.update(
            kw.strip().lower()
            for cell in df[col].dropna()
            for kw in cell.split(";")
            if " " in kw                         # frases de ≥ 2 palabras
        )
PHRASES_RAW.extend(sorted(kw_phrases))

# ───────────── 7 · Compilar patrones freeze
REGEX_PATTERNS = []
for ph in PHRASES_RAW:
    ph_lc = ph.lower()
    if ph_lc.endswith("*"):                          # wildcard de prefijo
        prefix = re.escape(ph_lc[:-1].strip())
        REGEX_PATTERNS.append(
            (re.compile(rf"\b{prefix}\w*\b", flags=re.I),
             lambda m: m.group(0).replace(" ", "_"))
        )
    else:                                            # coincidencia exacta
        REGEX_PATTERNS.append(
            (re.compile(re.escape(ph_lc), flags=re.I),
             ph_lc.replace(" ", "_"))
        )

def freeze_phrases(text: str) -> str:
    if not isinstance(text, str): return ""
    txt = text.lower()
    for pat, repl in REGEX_PATTERNS:
        txt = pat.sub(repl, txt)
    return txt

# ───────────── 8 · Funciones utilitarias
def lang_of(text: str) -> str:
    try: return detect(text[:400])
    except LangDetectException:
        return "es" if any(c in "áéíóúñ" for c in text) else "en"
# palabras que se deben conservar tal cual
LEMMA_EXCEPTIONS = {"data", "media", "news", "series", "species"}


def lematizar(texto: str, modelo) -> list[str]:
    return [
        tok.text.lower() if tok.lemma_.lower() in LEMMA_EXCEPTIONS else tok.lemma_
        for tok in modelo(texto)
        if tok.pos_ in {"NOUN", "PROPN", "ADJ"}
           and tok.is_alpha and len(tok) > 2
           and tok.lemma_.lower() not in STOP
    ]
def keywords_a_lista(celda: str) -> list[str]:
    if not isinstance(celda, str): return []
    return [unidecode.unidecode(k.strip().lower()) for k in celda.split(";") if k.strip()]

# ───────────── 9 · Procesar filas
def fila_a_tokens(row) -> str:
    texto  = unidecode.unidecode(freeze_phrases(row["TextoFull"]))
    modelo = nlp_es if lang_of(texto).startswith("es") else nlp_en
    lemas  = lematizar(texto, modelo)
    kw_a   = keywords_a_lista(row.get("Author Keywords"))
    kw_i   = keywords_a_lista(row.get("Index Keywords"))
    tokens = lemas + kw_a + kw_i
    tokens = [tok.replace("_", " ") for tok in tokens]   # quita “_”
    tokens = list(dict.fromkeys(tokens))                 # deduplicar
    return ";".join(tokens)

print("⚙️  Procesando filas…")
df["ResumenTokens"] = df.apply(fila_a_tokens, axis=1)

df['Author Keywords'] = df["ResumenTokens"]

def contar_unicos(col, df_input):
    # Drop NA, split por “;”, strip y filtrar vacíos
    all_keywords = (
        df_input[col]
        .dropna()
        .str.split(';')
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
    )
    return all_keywords.unique().shape[0]

# Conteo único antes de normalizar
print("Antes de normalizar:")
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")
# --- 2) Definir los scorers disponibles ---
SCORERS = {
    'ratio':       fuzz.ratio,
    'partial':     fuzz.partial_ratio,
    'token_sort':  fuzz.token_sort_ratio,
    'token_set':   fuzz.token_set_ratio,
    'WRatio':   fuzz.WRatio
    
}
def to_singular(term: str) -> str:
    tokens = term.split()
    if not tokens:
        return term
    last = tokens[-1]
    if re.match(r".{4,}ies$", last):          # studies → study
        last = last[:-3] + "y"
    elif re.match(r".{3,}s$", last) and not last.endswith("ss"):
        last = last[:-1]                      # structures → structure
    tokens[-1] = last
    return " ".join(tokens)
# --- 3) Función de agrupamiento con bandera “method” ---
def agrupar_rápido(
    cell: str,
    vocab: list[str],
    thresh: int = 92,
    method: str = "WRatio"
) -> str:
    if not isinstance(cell, str) or not cell.strip():
        return ""

    scorer = SCORERS.get(method, fuzz.WRatio)
    seen, out = set(), []

    for raw in [t.strip().lower() for t in cell.split(';') if t.strip()]:
        term = to_singular(raw)

        # mejor coincidencia en todo el vocabulario (sin el mismo término)
        best = process.extractOne(
            query   = term,
            choices = [to_singular(v) for v in vocab if to_singular(v) != term],
            scorer  = scorer,
            score_cutoff = thresh          # ≤ thresh no hay emparejamiento
        )

        if best:
            match, score = best[0], best[1]
            # elige la más corta entre term y match
            elegido = match if len(match) < len(term) else term
        else:
            elegido = term

        if elegido not in seen:
            seen.add(elegido)
            out.append(elegido)

    return ';'.join(out)
for col in ["Author Keywords", "Index Keywords"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)
# --- 5) Construye vocabulario global ---
def construir_vocab(df, col):
    return sorted({
        kw.strip().lower()
        for cell in df[col]
        for kw in cell.split(';') if kw.strip()
    })


vocab_global = sorted(
    set(construir_vocab(df,'Author Keywords') +
        construir_vocab(df,'Index Keywords'))
)
# Token sort ratio
df['Author Keywords'] = df['Author Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=95, method='WRatio'))
#df['Index Keywords'] = df['Index Keywords'].apply(lambda c: agrupar_rápido(c, vocab_global, thresh=70, method='WRatio'))

print("despúes de normalizar:")
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")
# eliminar las columnas 
df.drop('TextoFull', axis=1, inplace=True)
df.drop('ResumenTokens', axis=1, inplace=True)
# ───────────── 10 · Guardar
print("💾 Guardando", OUT_FILE)
df.to_csv(OUT_FILE, index=False)
print("✅ Terminado sin errores.")