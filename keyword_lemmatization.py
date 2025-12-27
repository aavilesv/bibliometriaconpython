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
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")
INPUT_CSV   = r"G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv"
OUTPUT_CSV  = r"G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv"
CHANGE_LOG  = r"G:\\Mi unidad\\papersPreprocesseloh.csv"

KW_COLS     = ["bothKeywords"]

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
 
    # ===== SIGLAS / ACRÓNIMOS =====
    "AI", "ML", "NLP", "ICT", "ICTs", "LLM", "LLMs",
    "AIEd", "GenAI", "GPT", "GPT-3", "UTAUT", "UTAUT2",
    "TAM", "PLS-SEM", "SEM", "TF-IDF", "LSTM",
    "KNN", "SVM", "XgBoost", "WEKA",

    # ===== MODELOS / ALGORITMOS =====
    "Random Forest", "Random Forests",
    "Support Vector Machine", "Support Vector Machines",
    "Decision Tree", "Decision Trees",
    "Neural Networks", "Artificial Neural Networks",
    "Convolutional Neural Network", "Convolutional Neural Networks",
    "Deep Neural Network",
    "Naïve Bayes",
    "Adaptive Boosting",
    "Genetic Algorithm", "Genetic Algorithms",
    "Reinforcement Learning",
    "Self-supervised Learning",
    "Supervised Learning",
    "Unsupervised Learning",
    "Transfer Learning",
    "Federated Learning",
    "Contrastive Learning",
    "Clustering",
    "Classification",
    "Regression",
    "Logistic Regression",

    # ===== FRAMEWORKS / TEORÍAS =====
    "Technology Acceptance Model",
    "Unified Theory of Acceptance and Use of Technology",
    "Theory of Planned Behavior",
    "Task-Technology Fit",
    "SELF-DETERMINATION THEORY",
    "Community of Inquiry",
    "TPACK",

    # ===== HERRAMIENTAS / PLATAFORMAS =====
    "ChatGPT", "Chat GPT",
    "OpenAI",
    "VOSviewer",
    "SCOPUS",
    "MOOCs",
    "LMS",
    "Learning Management System",
    "Learning Management Systems",
    "Internet of Things", "IoT",

    # ===== ÁREAS / CONCEPTOS TÉCNICOS FIJOS =====
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Generative Artificial Intelligence",
    "Large Language Models",
    "Natural Language Processing",
    "Educational Data Mining",
    "Learning Analytics",
    "Explainable Artificial Intelligence",
    "Computer Vision",
    "Sentiment Analysis",
    "Text Mining",
    "Data Mining",
    "Predictive Analytics",
    "Business Intelligence",

    # ===== EDUCACIÓN / SISTEMAS ESPECÍFICOS =====
    "Higher Education",
    "Higher Education Institutions",
    "Universities",
    "Education 4.0",
    "Industry 4.0",
    "Industry 5.0",
    "Intelligent Tutoring Systems",
    "Virtual Reality",
    "Augmented Reality",
    "Blockchain",
    "Metaverse",

    # ===== ESTÁNDARES / MÉTRICAS =====
    "Accuracy",
    "Quality Assurance",
    "Quality Control",

    # ===== NOMBRES GEOGRÁFICOS / PROPIOS =====
    "China", "India", "United States", "United Kingdom",
    "Saudi Arabia", "Australia", "Canada", "Germany",
    "Romania", "Thailand", "e-learning",
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