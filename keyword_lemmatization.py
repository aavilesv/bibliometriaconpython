# -*- coding: utf-8 -*-
"""
02 - Lematización y Canonización (MODO SEGURO)
- Lematización spaCy (inteligente)
- Protege siglas científicas (REDD+, R&D, CO2)
- Conserva números y años (2030, 4.0)
- Elimina duplicados finales
"""

import re, time
import sys
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import pandas as pd
from unidecode import unidecode
import spacy

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

INPUT_CSV   = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopuscorreci.csv"
OUTPUT_CSV  = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusbloque1replacelematizar.csv"
CHANGE_LOG  = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\02_lemmatize_canonize_log.csv"

KW_COLS     = ["Author Keywords", "Index Keywords"]

# Cargar Spacy (asegúrate de tenerlo instalado: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    print("❌ ERROR: No tienes el modelo de Spacy.")
    print("👉 Ejecuta en terminal: python -m spacy download en_core_web_sm")
    sys.exit()

# ==========================================
# 2. LISTAS DE EXCEPCIÓN Y REGLAS
# ==========================================

# Mantenemos tu lista completa, pero formateada correctamente para evitar errores
EXCEPTION_PHRASES = {
    # --- TÉRMINOS GENERALES ---
    "data", "e-leadership", "digital leadership", "virtual leadership", "transformational leadership",
    
    # --- ENTIDADES / PAÍSES ---
    "china", "united states", "brazil", "canada", "india", "australia",
    "latin america", "european union", "africa", "arctic", "new zealand",
    "pacific islands", "amazonia", "ecuador", "asia", "global south",
    "germany", "chile", "peru", "indonesia", "oecd countries",

    # --- ORGANISMOS ---
    "united nations", "united nations framework convention on climate change",
    "unfccc", "ipcc", "ipbes", "intergovernmental panel", "unep", "world polity",

    # --- TRATADOS ---
    "paris agreement", "kyoto protocol", "montreal protocol",
    "convention on biological diversity", "stockholm convention", "agenda 2030",

    # --- SIGLAS (Protegidas de lematización) ---
    "redd+", "cdm", "sdgs", "sdg", "co2", "ghg", "ngos", "oecd",

    # --- CONCEPTOS JURÍDICOS ---
    "international environmental law", "environmental law", "administrative law",
    "procedural justice", "human rights", "indigenous rights", "rights of nature",
    "right to a healthy environment", "soft law", "hard law", 
    "public policy", "environmental impact assessment",

    # --- MODELOS ---
    "porter hypothesis", "environmental kuznets curve", "stirpat model",
    "q methodology", "difference in differences", "panel data",
    "time series", "conceptual framework",

    # --- GOBERNANZA ---
    "global environmental governance", "multilevel governance", "adaptive governance",
    "collaborative governance", "polycentric governance", "climate governance",
    "environmental governance", "transnational governance", "urban governance",
    "network governance", "co-management", "co production", "co-production",

    # --- CIENCIA ---
    "science-policy interface", "science-policy", "literature review",
    "boundary organizations", "epistemic communities",

    # --- AMBIENTAL (NO LEMATIZAR) ---
    "climate change", "climate change adaptation", "climate change mitigation",
    "climate adaptation", "climate migration", "climate justice", "climate risk",
    "global change", "global climate", "air pollution",
    "carbon emissions", "carbon dioxide emissions", "carbon footprint",
    "greenhouse gas emissions", "biodiversity", "ecosystem services",
    "deforestation", "renewable energy", "sustainable development",
    "sustainable development goals", "ecosystem based management",
    "land use change", "nature based solutions", "marine biodiversity",
    "ocean acidification",

    # --- ECONOMÍA ---
    "carbon market", "carbon tax", "carbon intensity", "emissions trading",
    "emissions trading scheme", "cap-and-trade", "market environmentalism",
    "financial performance", "foreign direct investment",

    # --- TEORÍA ---
    "political ecology", "neoliberalism", "anthropocene", "authoritarian environmentalism",
    "ecological modernization", "social learning", "collective action",
    "institutional theory", "resilience thinking",

    # --- DERECHO INTERNACIONAL ---
    "law of the sea", "international law", "international cooperation",
    "international agreement", "international environmental agreements",
    "international governance", "international legal framework", "international trade",

    # --- VARIOS ---
    "big data", "boundary objects", "convention", "partition", 
    "urban areas", "urban sustainability", "smart cities", "food security",
    "forest management", "environmental justice", "environmental protection"
}

# Palabras que Spacy a veces identifica mal como verbos o plurales y no debería tocar
EXCEPTION_NOUNS = {"autism spectrum disorder", "data", "media"}

# Diccionario UK -> US
BRIT_US = {
    "behaviour": "behavior", "behaviours": "behavior",
    "organisation": "organization", "organisations": "organizations",
    "programme": "program", "programmes": "programs",
    "centre": "center", "centres": "centers",
    "analyse": "analyze", "analysed": "analyzed", "modelling": "modeling"
}

# Mapeo canónico de frases (corrección forzada)
PHRASE_CANON = [
    (r"artificial neural network", "neural network"), 
    (r"neural networks", "neural network")
]

# Mapeo canónico de tokens individuales
TOKEN_CANON = {}

# ==========================================
# 3. FUNCIONES SEGURAS
# ==========================================

# Regex permitida: Letras, números, guiones, espacios Y SÍMBOLOS (+ . % & /)
SAFE_REGEX = re.compile(r"[^a-z0-9'\-\s\+\.\%\&\/]") 

RE_DOTTED = re.compile(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b')
RE_SINGLE = re.compile(r'^[a-z]$')
RE_DIGITS = re.compile(r'^\d+$')
_PH_MARK  = "\uFFF1"

# Pre-compilar patrones para velocidad
EXC_PATTERNS = [(re.compile(rf"\b{re.escape(p)}\b"), p) for p in sorted(EXCEPTION_PHRASES, key=len, reverse=True)]

def limpieza_suave(s: str) -> str:
    """Limpieza que respeta símbolos científicos"""
    if not isinstance(s, str): return ""
    s = unidecode(s.lower())
    # AQUÍ ESTÁ LA CLAVE: No borramos símbolos científicos
    s = SAFE_REGEX.sub(" ", s) 
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mask_exception_phrases(text: str):
    """Protege las frases de la lista EXCEPTION_PHRASES"""
    mapping, idx = {}, 0
    for pat, phrase in EXC_PATTERNS:
        if pat.search(text):
            key = f"{_PH_MARK}{idx}"
            mapping[key] = phrase
            text = pat.sub(key, text); idx += 1
    return text, mapping

def unmask_exception_phrases(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def aplicar_mapa(texto: str, mapping: dict) -> str:
    """Aplica diccionario UK->US"""
    toks = texto.split()
    return " ".join([mapping.get(t, t) for t in toks])

def canonize_phrases(text: str) -> str:
    for k, v in PHRASE_CANON:
        text = re.sub(rf"\b{k}\b", v, text)
    return text

def filtrar_basura(term: str) -> str:
    """Filtra basura pero permite números y letras científicas"""
    toks = []
    allowed_single = {'a', 'i', 't', 'b', 'n', 'x', 'y', 'z', 'p'} # t-test, x-ray
    for t in term.split():
        # Permitir si es número (ej: 2030, 4.0) O si es letra permitida
        if RE_DIGITS.match(t):  
            toks.append(t)
            continue
        if RE_SINGLE.match(t) and t not in allowed_single: 
            continue
        toks.append(t)
    return " ".join(toks).strip()

def lemmatize_spacy(texto: str) -> str:
    """Lematización inteligente con Spacy"""
    doc = nlp(texto)
    out = []
    for tok in doc:
        # No lematizamos si parece una sigla (todo mayus, aunque aquí llega minúscula)
        # o si es un pronombre posesivo 's
        if tok.text in ("'s", "'"): continue
        
        if tok.is_space or tok.is_punct: 
            # Mantenemos puntuación interna relevante si sobrevivió la limpieza
            if tok.text in ['+', '&', '%']: out.append(tok.text)
            continue
            
        orig, lem = tok.text, tok.lemma_
        
        # Reglas de excepción para no lematizar incorrectamente
        if tok.tag_ == "VBG": # Gerundios a veces mejor dejarlos (learning vs learn)
            out.append(orig)
        elif tok.pos_ == "NOUN" and orig in EXCEPTION_NOUNS:
            out.append(orig)
        elif lem == "-PRON-":
            out.append(orig)
        else:
            out.append(lem)
            
    return " ".join(out).strip()

@lru_cache(maxsize=200_000)
def normalize_single_keyword(kw: str) -> str:
    """Proceso para UNA sola palabra"""
    # 1. Limpieza suave
    kw_clean = limpieza_suave(kw)
    if not kw_clean: return kw # Devolver original si se borró todo por error
    
    # 2. Enmascarar frases protegidas (China, United Nations...)
    kw_masked, exc_map = mask_exception_phrases(kw_clean)

    # 3. Lematizar lo NO protegido
    # Si el texto es TODO una máscara, saltamos lematización
    if not kw_masked.startswith(_PH_MARK):
        kw_lem = lemmatize_spacy(kw_masked)
    else:
        kw_lem = kw_masked

    # 4. UK -> US
    kw_us = aplicar_mapa(kw_lem, BRIT_US)

    # 5. Canonización (Neural networks -> neural network)
    kw_canon = canonize_phrases(kw_us)

    # 6. Restaurar frases protegidas
    kw_final = unmask_exception_phrases(kw_canon, exc_map)

    # 7. Filtro final suave
    result = filtrar_basura(kw_final)
    
    # SALVAVIDAS: Si quedó vacío, devolvemos la versión limpia básica
    return result if result else kw_clean

def normalize_cell(cell: str) -> str:
    if not isinstance(cell, str): return ""
    raw = [t.strip() for t in cell.split(';') if t.strip()]
    out = []
    seen = set() # ¡IMPORTANTE! Eliminar duplicados
    
    for term in raw:
        norm = normalize_single_keyword(term)
        if norm:
            norm_lower = norm.lower()
            if norm_lower not in seen:
                seen.add(norm_lower)
                out.append(norm)
                
    return "; ".join(out)

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    t0 = time.perf_counter()
    
    print(f"📂 Leyendo CSV: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV).fillna("")
    except FileNotFoundError:
        print("❌ ERROR: Archivo no encontrado."); sys.exit()

    log_rows = []
    
    # Recuento previo
    print("\n📊 Estadísticas ANTES:")
    for c in KW_COLS:
        if c in df.columns:
            print(f"  - {c}: {df[c].dropna().nunique()} filas únicas")

    # Procesar columnas
    for col in KW_COLS:
        if col not in df.columns: continue
        print(f"\n🚀 Lematizando columna: '{col}' ...")
        
        before = df[col].astype(str)
        after  = before.apply(normalize_cell)
        
        changed = (before != after)
        n_changed = changed.sum()
        print(f"   -> Celdas modificadas: {n_changed}")

        if n_changed > 0:
            log_rows.append(pd.DataFrame({
                "row_index": df.index[changed],
                "column": col,
                "before": before[changed],
                "after":  after[changed]
            }))
        
        df[col] = after

    # Guardar
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n💾 CSV guardado: {OUTPUT_CSV}")

    if log_rows:
        pd.concat(log_rows).to_csv(CHANGE_LOG, index=False, encoding="utf-8-sig")
        print(f"📝 Log guardado: {CHANGE_LOG}")
    else:
        print("📝 Sin cambios sustanciales.")

    print(f"\n⏱️ Tiempo total: {time.perf_counter() - t0:.2f} s")