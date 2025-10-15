# -*- coding: utf-8 -*-
"""
===============================================================
PIPELINE DE LIMPIEZA/UNIFICACIÓN (Title + Abstract) Y KEYWORDS
===============================================================

Se respetan TODAS las columnas originales del CSV (NO se elimina nada).
Se AÑADEN estas columnas:

- text_raw      -> Title + ". " + Abstract (limpieza básica)
- text_norm     -> Normalización (minúsculas, sin acentos, variantes)
- text_clean    -> Versión lematizada (sin stopwords)
- Keywords Unified -> Fusión normalizada (Index + Author Keywords)

Archivos exportados:
- CSV completo (para Excel, Power BI, VOSviewer)
- Parquet completo (para algoritmos pesados en Python)
"""

import re, regex
import unicodedata
import pandas as pd
from unidecode import unidecode
from pathlib import Path
import spacy
from itertools import combinations
import logging

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------
INPUT  = r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replacelematizar.csv"
OUTPUT = str(Path(INPUT).with_name("datawos_scopusbloque1_cleanfinal.csv"))
OUTPUT_PARQ = str(Path(INPUT).with_name("datawos_scopusbloque1_cleanfinal.parquet"))

COL_TITLE = "Title"
COL_ABS   = "Abstract"
COL_IDXKW = "Index Keywords"
COL_AUTHK = "Author Keywords"

SPACY_MODEL = "en_core_web_sm"

# -----------------------------
# FRASES PROTEGIDAS
# -----------------------------
EXCEPTION_PHRASES = [
    "data", "e-leadership", "digital leadership", "virtual leadership",
    "transformational leadership", "industry 4 0", "digital transformation",
    "higher education", "covid-19", "artificial intelligence", "machine learning",
    "bibliometric analysis", "systematic review", "knowledge management",
    "pls-sem", "organizational performance", "leadership theory", "leadership style",
    "innovation", "education", "performance", "trust", "motivation", "resilience",
    "organization", "management", "technology-enhanced learning",
    "higher education institutions (heis)", "tertiary education", "student leadership"
]

# -----------------------------
# VARIANTES A NORMALIZAR
# -----------------------------
VARIANTS_MAP = {
    r"\be[\-\s]?lead(er(ship)?|ers?)\b": "e-leadership",
    r"\bindustry\s*4[\.\s]?0\b": "industry 4.0",
    r"\bindustry\s*5[\.\s]?0\b": "industry 5.0",
    r"\bcovid[\s\-]?19\b": "covid-19",
    r"\bleader[–-]member exchange\b": "leader–member exchange",
    r"\butaut\s*3\b": "UTAUT3",
    r"\bvisual\s*simultaneous\s*localization\s*and\s*mapp(ing|ings)\b": "Visual SLAM",
    r"\bai\b": "AI", r"\bnlp\b": "NLP", r"\bhmm\b": "HMM", r"\bsvm\b": "SVM",
    r"\blda\b": "LDA", r"\biot\b": "IoT"
}

STOP_EXTRA = {
    "et","al","figure","fig","table","tables","result","results","study","paper","using","use",
    "based","may","however","therefore","conclusion","methods","method","approach","analysis",
    "findings","implications","limitations","purpose","aim","objective","objectives","background",
    "introduction","discussion","contribution","novel","new"
}

CLEAN_PATTERNS = [
    r"©?\s*\d{4}.*elsevier.*", r"all rights reserved.*", r"rights reserved.*",
    r"springer nature.*", r"mdpi.*licensee.*", r"creativecommons.*license.*",
    r"\bexc\b", r"\belsevier\b"
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def basic_clean(s: str) -> str:
    s = nfkc(s).replace("\u00ad", "").replace("\xa0", " ")
    return regex.sub(r"\s+", " ", s).strip()

def apply_variants(text: str) -> str:
    for rx, rep in VARIANTS_MAP.items():
        text = regex.sub(rx, rep, text, flags=regex.IGNORECASE | regex.UNICODE)
    return text

def compile_exception_patterns(phrases):
    ordered = sorted(phrases, key=lambda x: len(x), reverse=True)
    patterns = []
    for p in ordered:
        esc = regex.escape(p).replace(r"\ ", r"\s+").replace(r"\-", r"[\-–]?")
        patt = regex.compile(rf"(?i)\b{esc}\b")
        patterns.append((p, patt))
    return patterns

EXC_PATTERNS = compile_exception_patterns(EXCEPTION_PHRASES)

def protect_exceptions(text: str) -> str:
    def placeholder(phrase):
        return "__EXC__" + re.sub(r"\W+", "_", phrase.strip().lower()) + "__"
    for phrase, patt in EXC_PATTERNS:
        text = patt.sub(placeholder(phrase), text)
    return text

def restore_exceptions(text: str) -> str:
    for phrase, _ in EXC_PATTERNS:
        ph = "__EXC__" + re.sub(r"\W+", "_", phrase.strip().lower()) + "__"
        text = text.replace(ph, phrase)
    return text

def remove_noise_phrases(text: str) -> str:
    if not text:
        return text
    for pat in CLEAN_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return basic_clean(text)

def normalize_text_pipeline(s: str) -> str:
    s = basic_clean(s)
    s = apply_variants(s)
    s = protect_exceptions(s)
    s = s.lower()
    s = unidecode(s)
    s = basic_clean(s)
    s = restore_exceptions(s)
    return s

def load_spacy(model: str):
    return spacy.load(model, disable=["ner","textcat"])

def build_stopwords(nlp, extra):
    stop = set(w.lower() for w in nlp.Defaults.stop_words)
    stop |= set(x.lower() for x in extra)
    stop -= {"no","not","without","vs"}
    return stop

def normalize_keywords_cell(val: str):
    if val is None:
        return []
    s = basic_clean(str(val))
    if not s:
        return []
    parts = re.split(r";|\|", s)
    out, seen = [], set()
    for p in parts:
        kw = basic_clean(p)
        if not kw:
            continue
        kw = apply_variants(kw)
        kw = protect_exceptions(kw)
        kw = restore_exceptions(kw)
        kkey = kw.lower()
        if kkey not in seen:
            out.append(kw)
            seen.add(kkey)
    return out

# ===============================================================
# BLOQUE PRINCIPAL (para evitar errores de multiprocessing)
# ===============================================================
def main():
    logging.info("Loading CSV…")
    df = pd.read_csv(INPUT, encoding="utf-8", dtype=str)

    original_cols = list(df.columns)

    for c in [COL_TITLE, COL_ABS]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in the CSV.")
    for c in [COL_IDXKW, COL_AUTHK]:
        if c not in df.columns:
            df[c] = ""

    df[COL_TITLE] = df[COL_TITLE].fillna("").astype(str)
    df[COL_ABS]   = df[COL_ABS].fillna("").astype(str)

    # 1️⃣ Unir y limpiar
    df["text_raw"] = (df[COL_TITLE].map(basic_clean) + ". " + df[COL_ABS].map(basic_clean)).str.strip()
    df["text_raw"] = df["text_raw"].map(remove_noise_phrases)

    # 2️⃣ Normalización
    df["text_norm"] = df["text_raw"].map(normalize_text_pipeline)

    # 3️⃣ Lematización
    logging.info("Loading spaCy model…")
    nlp = load_spacy(SPACY_MODEL)
    STOP = build_stopwords(nlp, STOP_EXTRA)

    def _prep_for_lemma(s: str) -> str:
        s2 = protect_exceptions(s)
        return re.sub(r"__EXC__([a-z0-9_]+)__", r"\1", s2)

    _texts = df["text_norm"].map(_prep_for_lemma).tolist()
    docs = list(nlp.pipe(_texts, batch_size=200, n_process=1))  # 👈 evita error en Windows

    clean_list = []
    for doc in docs:
        toks = []
        for t in doc:
            if t.is_space or t.is_punct or t.like_num or t.like_url or t.like_email:
                continue
            lemma = t.lemma_.lower().strip()
            if lemma in STOP or len(lemma) < 2:
                continue
            toks.append(lemma)
        clean_list.append(" ".join(toks))

    df["text_clean"] = [basic_clean(remove_noise_phrases(restore_exceptions(
                            re.sub(r"_", " ", txt)))) for txt in clean_list]

    # 4️⃣ Fusionar Keywords
    kw_merged = []
    for idxkw, authkw in zip(df[COL_IDXKW].fillna(""), df[COL_AUTHK].fillna("")):
        li = normalize_keywords_cell(idxkw) + normalize_keywords_cell(authkw)
        seen, uniq = set(), []
        for x in li:
            k = x.lower()
            if k not in seen:
                uniq.append(x)
                seen.add(k)
        kw_merged.append("; ".join(uniq))
    df["Keywords Unified"] = kw_merged

    # 5️⃣ Añadir métricas QC
    df["year"] = pd.to_datetime(df.get("Year", ""), errors="coerce").dt.year
    df["n_tokens_raw"]   = df["text_raw"].str.split().str.len()
    df["n_tokens_clean"] = df["text_clean"].str.split().str.len()
    df["kw_count"]       = df["Keywords Unified"].apply(lambda s: 0 if not s else s.count(";") + 1)
    df["has_doi"]        = df.get("DOI", pd.Series([""]*len(df))).astype(str) \
                              .str.contains(r"10\.\d{4,9}/", case=False, na=False)

    # 6️⃣ Exportar archivos
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    try:
        df.to_parquet(OUTPUT_PARQ, index=False)
    except Exception as e:
        logging.warning(f"No se pudo escribir Parquet: {e}")

    logging.info(f"✅ Export CSV: {OUTPUT}")
    logging.info(f"✅ Export Parquet: {OUTPUT_PARQ}")
    logging.info("Pipeline terminado correctamente ✓")

# Requerido en Windows (evita RuntimeError)
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
