# -*- coding: utf-8 -*-
"""
01 - Corrección contextual de Author/Index Keywords (IN-PLACE)
- Limpieza básica
- Protección de acrónimos/tecnicismos y FRASES-EXCEPCIÓN
- Corrección ortográfica CONTEXTUAL (JamSpell -> contextualSpellCheck -> SymSpell)
- Filtro básico de basura
- Log de cambios (antes/después)
"""

import re, time
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import pandas as pd
from unidecode import unidecode
from rapidfuzz.distance import Levenshtein

# ====== Rutas ======
INPUT_CSV   = r"G:\Mi unidad\2025\master Avila Coello Alex Armando\data\datawos_scopusreemplazar.csv"
OUTPUT_CSV  = r"G:\Mi unidad\2025\master Avila Coello Alex Armando\data\datawos_scopuscorreci.csv"
CHANGE_LOG  = r"G:\Mi unidad\2025\master Avila Coello Alex Armando\data\01_corrections_log.csv"
KW_COLS     = ["Author Keywords", "Index Keywords"]

# ====== Parámetros ======
PRESSURE = "high"   # "high" | "medium" | "low"

# (Opcional) JamSpell
JAMSPELL_MODEL_PATH = r"G:\Mi unidad\2025\master karla mora\new article scopus\models\jamspell\en.bin"

# ====== Frases-excepción (no tocar) - en minúsculas ======
EXCEPTION_PHRASES = {
    # Ambientes/prácticas
    "home literacy environment", "home numeracy environment", "home learning environment",
    "shared reading", "dialogic reading", "guided play", "serve and return",
    "responsive caregiving", "home-based", "home visit", "home visits", "home visiting",
    "house calls", "child-directed speech", "parent-child interaction", "parent-child interactions",

}

# ====== Acrónimos/tecnicismos a proteger ======
PROTECT_TOKENS = {

}

# ====== Dependencias de corrección ======
from symspellpy import SymSpell, Verbosity
import pkg_resources
import spacy

# SymSpell (fallback)
MAX_EDIT_DISTANCE = 2
PREFIX_LENGTH     = 7
sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=PREFIX_LENGTH)
freq_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym.load_dictionary(freq_path, 0, 1)

# JamSpell (opcional)
JAMSPELL_READY = False
try:
    from jamspell import TSpellCorrector
    _jam = TSpellCorrector()
    JAMSPELL_READY = _jam.LoadLangModel(JAMSPELL_MODEL_PATH)
except Exception:
    JAMSPELL_READY = False

# contextualSpellCheck (opcional)
CSC_READY = False
try:
    import contextualSpellCheck
    _nlp_csc = spacy.load("en_core_web_trf")
    contextualSpellCheck.add_to_pipe(_nlp_csc)
    CSC_READY = True
except Exception:
    CSC_READY = False

# ====== Utilidades ======
RE_DOTTED = re.compile(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b')
RE_SINGLE = re.compile(r'^[a-z]$')
RE_DIGITS = re.compile(r'^\d+$')
_PH_MARK  = "\uFFF1"
_PROT_MARK= "\uFFF0"
TECH_RE   = re.compile(r"[0-9_/]|(^[a-z]+[A-Z][a-z]*$)")

def undot_acronyms(s: str) -> str:
    return RE_DOTTED.sub(lambda m: m.group(0).replace('.', ''), s)

def limpieza_basica(s: str) -> str:
    if not isinstance(s, str): return ""
    s = undot_acronyms(s)
    s = unidecode(s.lower())
    s = s.replace('(', ' ').replace(')', ' ')
    s = re.sub(r"[^a-z0-9'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Frases excepción (más largas primero)
EXC_PATTERNS = [(re.compile(rf"\b{re.escape(p)}\b"), p) for p in sorted(EXCEPTION_PHRASES, key=len, reverse=True)]

def mask_exception_phrases(text: str):
    mapping, idx = {}, 0
    for pat, phrase in EXC_PATTERNS:
        key = f"{_PH_MARK}{idx}"
        if pat.search(text):
            mapping[key] = phrase
            text = pat.sub(key, text)
            idx += 1
    return text, mapping

def unmask_exception_phrases(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def _needs_protect(tok: str) -> bool:
    if tok in PROTECT_TOKENS: return True
    if any(ch.isdigit() for ch in tok): return True
    if TECH_RE.search(tok): return True
    if "-" in tok and len(tok) <= 5: return True
    return False

def mask_protected_tokens(text: str):
    toks, mapping, out, idx = text.split(), {}, [], 0
    for t in toks:
        if _needs_protect(t):
            key = f"{_PROT_MARK}{idx}"
            mapping[key] = t
            out.append(key); idx += 1
        else:
            out.append(t)
    return " ".join(out), mapping

def unmask_protected_tokens(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def accept_change(orig: str, new: str) -> bool:
    if orig == new: return True
    d = Levenshtein.distance(orig, new)
    return d <= (4 if PRESSURE=="high" else 2 if PRESSURE=="low" else 3)

def correct_phrase_contextual(phrase: str) -> str:
    if not phrase.strip(): return phrase
    corrected = phrase

    if JAMSPELL_READY:
        try:
            tmp = _jam.FixFragment(phrase)
            if tmp: corrected = tmp
        except Exception: pass

    if (not JAMSPELL_READY or corrected == phrase) and CSC_READY:
        try:
            doc = _nlp_csc(phrase)
            if getattr(doc._, "performed_spellCheck", False):
                corrected = getattr(doc._, "outcome_spellCheck", phrase)
        except Exception: pass

    if corrected == phrase:
        res = sym.lookup_compound(phrase, max_edit_distance=MAX_EDIT_DISTANCE)
        if res: corrected = res[0].term

    # post filtro por presión
    o_toks, c_toks = phrase.split(), corrected.split()
    if len(o_toks) == len(c_toks):
        c2 = [c if accept_change(o, c) else o for o, c in zip(o_toks, c_toks)]
        corrected = " ".join(c2)
    return corrected

def filtrar_basura(term: str) -> str:
    toks = []
    for t in term.split():
        if RE_SINGLE.match(t):  continue
        if RE_DIGITS.match(t):  continue
        toks.append(t)
    return " ".join(toks).strip()

@lru_cache(maxsize=200_000)
def correct_single_keyword(kw: str) -> str:
    kw = limpieza_basica(kw)
    if not kw: return ""

    kw, exc_map  = mask_exception_phrases(kw)
    kw, prot_map = mask_protected_tokens(kw)

    kw = correct_phrase_contextual(kw)
    kw = filtrar_basura(kw)

    kw = unmask_protected_tokens(kw, prot_map)
    kw = unmask_exception_phrases(kw, exc_map)
    kw = filtrar_basura(kw)
    return kw


def correct_cell(cell: str) -> str:
    if not isinstance(cell, str):
        return ""
    raw = [t.strip() for t in cell.split(';') if t.strip()]
    out = []
    for term in raw:
        corr = correct_single_keyword(term)
        if corr:
            out.append(corr)   # ← ya no checamos 'seen'
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
        print(f"\nCorrigiendo: {col} ...")
        before = df[col].astype(str)
        after  = before.apply(correct_cell)
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
    print("📁 CSV corregido:", OUTPUT_CSV)
    print("="*60)
