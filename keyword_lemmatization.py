# -*- coding: utf-8 -*-
"""
02 - Lematización y canonización de Author/Index Keywords (IN-PLACE)
- Lematización spaCy
- UK→US / sinónimos
- Canonización de frases/tokens (VPN, IoT, QoS; neural networks→neural network)
- Filtro final
- Log de cambios (antes/después)
"""

import re, time
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import pandas as pd
from unidecode import unidecode
import spacy

# ====== Rutas ======
INPUT_CSV   = r"G:\Mi unidad\2025\master CASTRO CASTRO ARACELLY GISELLA\data\datawos_scopus.csv"
OUTPUT_CSV  = r"G:\Mi unidad\2025\master CASTRO CASTRO ARACELLY GISELLA\data\datawos_scopuslematizar.csv"
CHANGE_LOG  = r"G:\Mi unidad\2025\master CASTRO CASTRO ARACELLY GISELLA\data\logs\02_lemmatize_canonize_log.csv"
KW_COLS     = ["Author Keywords", "Index Keywords"]

# ====== Config ======
EXCEPTION_PHRASES = {
    # mismas excepciones que en el script 01
"data"
}

EXCEPTION_NOUNS = {"autism spectrum disorder"}

BRIT_US = {
    "behaviour": "behavior", "behaviours": "behavior",
    "organisation": "organization", "organisations": "organizations",
}

PHRASE_CANON = [
   
]

TOKEN_CANON = {

}

# ====== spaCy ======
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ====== Utils ======
RE_SINGLE = re.compile(r'^[a-z]$')
RE_DIGITS = re.compile(r'^\d+$')
_PH_MARK  = "\uFFF1"

EXC_PATTERNS = [(re.compile(rf"\b{re.escape(p)}\b"), p) for p in sorted(EXCEPTION_PHRASES, key=len, reverse=True)]

def limpieza_basica(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unidecode(s.lower())
    s = re.sub(r"[^a-z0-9'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mask_exception_phrases(text: str):
    mapping, idx = {}, 0
    for pat, phrase in EXC_PATTERNS:
        key = f"{_PH_MARK}{idx}"
        if pat.search(text):
            mapping[key] = phrase
            text = pat.sub(key, text); idx += 1
    return text, mapping

def unmask_exception_phrases(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def aplicar_mapa(texto: str, mapping: dict) -> str:
    for k, v in mapping.items():
        texto = re.sub(rf"\b{re.escape(k)}\b", v, texto)
    return texto

def canonize_phrases(text: str) -> str:
    for k, v in PHRASE_CANON:
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text

def canonize_tokens(text: str) -> str:
    toks = text.split()
    return " ".join([TOKEN_CANON.get(t, t) for t in toks])

def filtrar_basura(term: str) -> str:
    toks = []
    for t in term.split():
        if RE_SINGLE.match(t):  continue
        if RE_DIGITS.match(t):  continue
        toks.append(t)
    return " ".join(toks).strip()

def lemmatize_spacy(texto: str) -> str:
    doc = nlp(texto)
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct: continue
        orig, lem = tok.text, tok.lemma_
        if tok.tag_ == "VBG":
            out.append(orig)
        elif tok.pos_ == "NOUN" and orig in EXCEPTION_NOUNS:
            out.append(orig)
        else:
            out.append(lem)
    return " ".join(out).strip()

@lru_cache(maxsize=200_000)
def normalize_single_keyword(kw: str) -> str:
    kw = limpieza_basica(kw)
    if not kw: return ""

    # 1) No tocamos ortografía aquí (ya corregida en 01). Protegemos solo frases-excepción.
    kw, exc_map = mask_exception_phrases(kw)

    # 2) Lematizar
    kw = lemmatize_spacy(kw)

    # 3) UK→US / sinónimos
    kw = aplicar_mapa(kw, BRIT_US)

    # 4) Canonización
    kw = canonize_phrases(kw)
    kw = canonize_tokens(kw)

    # 5) Restaurar excepciones
    kw = unmask_exception_phrases(kw, exc_map)

    # 6) Filtro final
    kw = filtrar_basura(kw)
    return kw

def normalize_cell(cell: str) -> str:
    if not isinstance(cell, str):
        return ""
    raw = [t.strip() for t in cell.split(';') if t.strip()]
    out = []
    for term in raw:
        norm = normalize_single_keyword(term)
        if norm:
            out.append(norm)   # ← sin 'seen'
    return "; ".join(out)

# ====== MAIN ======
if __name__ == "__main__":
    t0 = time.perf_counter(); ts0 = datetime.now()
    df = pd.read_csv(INPUT_CSV).fillna("")
    Path(CHANGE_LOG).parent.mkdir(parents=True, exist_ok=True)

    log_rows = []
    print("Antes (recuento únicos):")
    for c in KW_COLS:
        if c in df.columns:
            nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
            print(f"  {c}: {nuniq}")

    for col in KW_COLS:
        if col not in df.columns: continue
        print(f"\nLematizando/canonizando: {col} ...")
        before = df[col].astype(str)
        after  = before.apply(normalize_cell)
        changed_mask = (before != after)
        if changed_mask.any():
            tmp = pd.DataFrame({
                "row_index": df.index[changed_mask],
                "column": col,
                "before": before[changed_mask].values,
                "after":  after[changed_mask].values
            })
            log_rows.append(tmp)
        df[col] = after

    print("\nDespués (recuento únicos):")
    for c in KW_COLS:
        if c in df.columns:
            nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
            print(f"  {c}: {nuniq}")

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    if log_rows:
        pd.concat(log_rows, ignore_index=True).to_csv(CHANGE_LOG, index=False, encoding="utf-8")
        print(f"\n📝 Log de cambios: {CHANGE_LOG}")
    else:
        print("\n📝 Sin cambios registrados.")

    t1 = time.perf_counter(); ts1 = datetime.now()
    print("\n" + "="*60)
    print("📅 Inicio:", ts0.strftime("%Y-%m-%d %H:%M:%S"))
    print("🕒 Fin   :", ts1.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"⏱️ Tiempo total de ejecución: {t1 - t0:.2f} s")
    print("📁 CSV final:", OUTPUT_CSV)
    print("="*60)
