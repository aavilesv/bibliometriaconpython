# -*- coding: utf-8 -*-
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

INPUT_CSV  = r"G:\\Mi unidad\\2025\\Master Italo Palacios\\articulo\\datawos_scopus.csv"
OUTPUT_CSV = r"G:\\Mi unidad\\2025\\Master Italo Palacios\\articulo\\datawos_scopus_keywords_norm.csv"
inicio_fecha_hora = datetime.now()
inicio_tiempo = time.perf_counter()
KW_COLS = ["Author Keywords", "Index Keywords"]  # Ajusta a tu CSV

# Palabras/exps a NO lematizar
EXCEPTION_NOUNS = {
    "data", "media", "criteria", "covid-19", "e-learning",
    "feedback physiological", "negative‑imaginary systems",
    "closed‑loop systems", "time‑varying control systems",
    "reverse transcriptase polymerase chain reaction"
}

# UK→US / sinónimos
BRIT_US = {
    "behaviour": "behavior", "behaviours": "behavior",
    "organisation": "organization", "organisations": "organizations",
    "colour": "color", "colours": "colors",
    "eco system": "ecosystem", "eco-systems": "ecosystems",
    "spatio-temporal": "spatiotemporal", "spatio temporal": "spatiotemporal",
    "urban forestry": "urban forest", "urban forests": "urban forest",
    "nature base solution": "nature-based solution",
    "contaminate land": "contaminated land",
    "contaminate site": "contaminated site",
    "carbon dioxide": "co2",
    # agrega lo que vayas encontrando
}

# Términos técnicos a “proteger” en el diccionario
CUSTOM_TERMS = [
    "ecosystem services", "ecosystem service", "urban forest",
    "urban tree canopy", "carbon sequestration", "climate change",
    "lda", "topic modeling", "ufvms", "green space", "spatiotemporal",
    "biodiversity conservation", "governance", "mortality",
    "air-pollution", "daily leisure", "low carbon behavior",
    "urban sustainability"
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

