# -*- coding: utf-8 -*-
"""
01 - Corrección contextual de Author/Index Keywords (OPTIMIZADO)
- Limpieza básica y normalización
- Protección de acrónimos/tecnicismos y FRASES-EXCEPCIÓN
- Corrección ortográfica EN CASCADA (JamSpell -> Contextual -> SymSpell)
- Eliminación de duplicados intra-celda
- Log de cambios detallado
"""

import re, time
import sys
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import pandas as pd
from unidecode import unidecode
from rapidfuzz.distance import Levenshtein
import pkg_resources
import spacy
from symspellpy import SymSpell, Verbosity

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================

# --- RUTAS DE ARCHIVOS (¡Verifica que existan!) ---
INPUT_CSV   = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopuseliminar2.csv"
OUTPUT_CSV  = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopuscorreci.csv"
CHANGE_LOG  = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\01_corrections_log.csv"

# --- RUTA JAMSPELL (Opcional) ---
JAMSPELL_MODEL_PATH = r"G:\Mi unidad\2025\master karla mora\new article scopus\models\jamspell\en.bin"

# --- COLUMNAS A PROCESAR ---
KW_COLS = ["Author Keywords", "Index Keywords"]

# --- PARÁMETROS DE CORRECCIÓN ---
# "high" = Muy agresivo (riesgo de cambiar tecnicismos). 
# "medium" = Balanceado (Recomendado). 
# "low" = Conservador.
PRESSURE = "medium" 

# ==========================================
# 2. LISTAS DE PROTECCIÓN
# ==========================================

# Frases que NO deben ser tocadas (en minúsculas)
EXCEPTION_PHRASES = {
    "china", "united states", "brazil", "canada", "india", "australia",
    "internet of things", "big data", "machine learning" # Ejemplos añadidos
}

# Acrónimos o tokens técnicos específicos a proteger
PROTECT_TOKENS = {
    "iot", "ai", "uav", "gis", "covid-19", "sars-cov-2"
}

# ==========================================
# 3. INICIALIZACIÓN DE MODELOS
# ==========================================

print("⏳ Cargando modelos de corrección...")

# A) SymSpell (Diccionario rápido)
MAX_EDIT_DISTANCE = 2
PREFIX_LENGTH     = 7
sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=PREFIX_LENGTH)
try:
    freq_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym.load_dictionary(freq_path, 0, 1)
except Exception as e:
    print(f"⚠️ Advertencia: No se pudo cargar diccionario SymSpell estándar. {e}")

# B) JamSpell (Contextual rápido)
JAMSPELL_READY = False
try:
    from jamspell import TSpellCorrector
    if Path(JAMSPELL_MODEL_PATH).exists():
        _jam = TSpellCorrector()
        JAMSPELL_READY = _jam.LoadLangModel(JAMSPELL_MODEL_PATH)
        print("✅ JamSpell cargado correctamente.")
    else:
        print("⚠️ No se encontró el modelo JamSpell en la ruta especificada.")
except ImportError:
    print("ℹ️ Librería JamSpell no instalada. Saltando...")
except Exception as e:
    print(f"⚠️ Error cargando JamSpell: {e}")

# C) ContextualSpellCheck (Spacy - Contextual profundo)
CSC_READY = False
try:
    import contextualSpellCheck
    # IMPORTANTE: Usamos 'en_core_web_sm' por velocidad. 
    # Si tienes GPU y quieres máxima precisión, cambia a 'en_core_web_trf'
    MODELO_SPACY = "en_core_web_sm" 
    
    print(f"⏳ Cargando Spacy ({MODELO_SPACY})...")
    _nlp_csc = spacy.load(MODELO_SPACY)
    contextualSpellCheck.add_to_pipe(_nlp_csc)
    CSC_READY = True
    print("✅ Spacy ContextualSpellCheck cargado.")
except OSError:
    print(f"❌ Error: No tienes descargado el modelo '{MODELO_SPACY}'.")
    print(f"   Ejecuta en terminal: python -m spacy download {MODELO_SPACY}")
except ImportError:
    print("ℹ️ Librería contextualSpellCheck o spacy no instalada.")
except Exception as e:
    print(f"⚠️ Error cargando ContextualSpellCheck: {e}")

# ==========================================
# 4. FUNCIONES DE UTILIDAD Y LIMPIEZA
# ==========================================

RE_DOTTED = re.compile(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b') # U.S.A. -> USA
RE_SINGLE = re.compile(r'^[a-z]$')      # Letras sueltas
RE_DIGITS = re.compile(r'^\d+$')        # Números solos
TECH_RE   = re.compile(r"[0-9_/]|(^[a-z]+[A-Z][a-z]*$)") # CamelCase o con números

_PH_MARK  = "\uFFF1" # Marcador temporal frases
_PROT_MARK= "\uFFF0" # Marcador temporal tokens

def undot_acronyms(s: str) -> str:
    """Elimina puntos en siglas (Ph.D. -> PhD)"""
    return RE_DOTTED.sub(lambda m: m.group(0).replace('.', ''), s)

def limpieza_basica(s: str) -> str:
    """Normalización inicial"""
    if not isinstance(s, str): return ""
    s = undot_acronyms(s)
    s = unidecode(s.lower()) # Quitar acentos y bajar a minúsculas
    s = s.replace('(', ' ').replace(')', ' ')
    # Permitimos letras, números, apóstrofes y guiones. Lo demás se va.
    s = re.sub(r"[^a-z0-9'\-\s]", " ", s) 
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Compilamos patrones de excepción para búsqueda rápida
EXC_PATTERNS = [(re.compile(rf"\b{re.escape(p)}\b"), p) for p in sorted(EXCEPTION_PHRASES, key=len, reverse=True)]

def mask_exception_phrases(text: str):
    """Protege frases enteras (ej: 'united states')"""
    mapping, idx = {}, 0
    for pat, phrase in EXC_PATTERNS:
        if pat.search(text):
            key = f"{_PH_MARK}{idx}"
            mapping[key] = phrase
            text = pat.sub(key, text)
            idx += 1
    return text, mapping

def unmask_exception_phrases(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def _needs_protect(tok: str) -> bool:
    """Decide si un token individual debe ser protegido del corrector"""
    if tok in PROTECT_TOKENS: return True
    if any(ch.isdigit() for ch in tok): return True # Tiene números (ej: 5g, cov2)
    if TECH_RE.search(tok): return True
    if "-" in tok and len(tok) <= 5: return True # Guiones cortos
    return False

def mask_protected_tokens(text: str):
    """Protege tokens individuales"""
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
    """Filtro de seguridad basado en distancia de Levenshtein"""
    if orig == new: return True
    d = Levenshtein.distance(orig, new)
    
    # Umbrales dinámicos según la presión configurada
    threshold = 4 if PRESSURE=="high" else 2 if PRESSURE=="low" else 3
    
    # Regla extra: Si la palabra es corta (<5 letras), toleramos menos cambios
    if len(orig) < 5:
        threshold = 1
        
    return d <= threshold

def correct_phrase_contextual(phrase: str) -> str:
    """Núcleo de corrección: JamSpell -> Spacy -> SymSpell"""
    if not phrase.strip(): return phrase
    corrected = phrase

    # 1. JamSpell
    if JAMSPELL_READY:
        try:
            tmp = _jam.FixFragment(phrase)
            if tmp: corrected = tmp
        except Exception: pass

    # 2. Spacy Contextual (Si JamSpell no cambió nada o no está)
    if (not JAMSPELL_READY or corrected == phrase) and CSC_READY:
        try:
            doc = _nlp_csc(phrase)
            if getattr(doc._, "performed_spellCheck", False):
                corrected = getattr(doc._, "outcome_spellCheck", phrase)
        except Exception: pass

    # 3. SymSpell (Fallback final palabra por palabra si sigue igual)
    if corrected == phrase:
        res = sym.lookup_compound(phrase, max_edit_distance=MAX_EDIT_DISTANCE)
        if res: corrected = res[0].term

    # 4. Verificación final (Safety Check)
    o_toks, c_toks = phrase.split(), corrected.split()
    # Si la longitud en palabras cambió drásticamente, sospechamos.
    # Si es igual longitud, verificamos palabra por palabra.
    if len(o_toks) == len(c_toks):
        c2 = [c if accept_change(o, c) else o for o, c in zip(o_toks, c_toks)]
        corrected = " ".join(c2)
    
    return corrected

def filtrar_basura(term: str) -> str:
    """Elimina tokens que son solo números o letras sueltas post-corrección"""
    toks = []
    for t in term.split():
        if RE_SINGLE.match(t) and t not in ('a', 'i'): continue # 'a' e 'i' son palabras válidas en inglés
        if RE_DIGITS.match(t):  continue
        toks.append(t)
    return " ".join(toks).strip()

@lru_cache(maxsize=200_000)
def correct_single_keyword(kw: str) -> str:
    """Pipeline completo para UNA sola palabra clave"""
    kw = limpieza_basica(kw)
    if not kw: return ""
    if len(kw) < 3: return kw # Ignorar cosas muy cortas

    # 1. Enmascarar
    kw, exc_map  = mask_exception_phrases(kw)
    kw, prot_map = mask_protected_tokens(kw)

    # 2. Corregir
    kw = correct_phrase_contextual(kw)
    kw = filtrar_basura(kw)

    # 3. Desenmascarar
    kw = unmask_protected_tokens(kw, prot_map)
    kw = unmask_exception_phrases(kw, exc_map)
    kw = filtrar_basura(kw)
    
    return kw

def correct_cell(cell: str) -> str:
    """Procesa una celda completa (varias keywords separadas por ;)"""
    if not isinstance(cell, str) or not cell.strip():
        return ""
    
    raw = [t.strip() for t in cell.split(';') if t.strip()]
    out = []
    seen = set() # Para evitar duplicados (ej: "AI; AI" -> "ai")

    for term in raw:
        corr = correct_single_keyword(term)
        if corr and corr not in seen:
            seen.add(corr)
            out.append(corr)
            
    return "; ".join(out)

# ==========================================
# 5. BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    t0 = time.perf_counter(); ts0 = datetime.now()
    
    # Crear carpetas si no existen
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Leyendo CSV: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV).fillna("")
    except FileNotFoundError:
        print("❌ ERROR: No se encontró el archivo CSV de entrada.")
        sys.exit()

    log_rows = []
    
    print("\n📊 Recuento de valores únicos ANTES:")
    for c in KW_COLS:
        if c in df.columns:
            nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
            print(f"  - {c}: {nuniq}")
        else:
            print(f"  ⚠️ Columna '{c}' no encontrada en el CSV.")

    # --- PROCESAMIENTO ---
    for col in KW_COLS:
        if col not in df.columns: continue
        print(f"\n🛠️  Corrigiendo columna: '{col}' ...")
        
        before = df[col].astype(str)
        # Aplicamos la corrección con barra de progreso simple (opcional visualmente)
        after  = before.apply(correct_cell)
        
        # Detección de cambios
        changed_mask = (before != after)
        count_changed = changed_mask.sum()
        print(f"   -> Filas modificadas: {count_changed}")

        if count_changed > 0:
            tmp = pd.DataFrame({
                "row_index": df.index[changed_mask],
                "column": col,
                "before": before[changed_mask].values,
                "after":  after[changed_mask].values
            })
            log_rows.append(tmp)
        
        df[col] = after

    print("\n📊 Recuento de valores únicos DESPUÉS:")
    for c in KW_COLS:
        if c in df.columns:
            nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
            print(f"  - {c}: {nuniq}")

    # --- GUARDADO ---
    print(f"\n💾 Guardando resultado en: {OUTPUT_CSV}")
    # Usamos utf-8-sig para mejor compatibilidad con Excel
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    if log_rows:
        log_df = pd.concat(log_rows, ignore_index=True)
        log_df.to_csv(CHANGE_LOG, index=False, encoding="utf-8-sig")
        print(f"📝 Log de cambios guardado en: {CHANGE_LOG}")
        print(f"   (Total cambios individuales: {len(log_df)})")
    else:
        print("\n📝 Sin cambios registrados. El archivo está limpio o la corrección no aplicó cambios.")

    t1 = time.perf_counter(); ts1 = datetime.now()
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print(f"⏱️  Tiempo total: {t1 - t0:.2f} segundos")
    print("="*60)