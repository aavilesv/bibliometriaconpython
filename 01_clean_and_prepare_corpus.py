# -*- coding: utf-8 -*-
"""
SCRIPT 01: LIMPIEZA Y PREPARACIÓN (NLP) - VERSIÓN COMPLETA
----------------------------------------------------------
Input: Tu archivo lematizado crudo.
Output: datawos_scopusbloque1_cleanfinal.csv
Mejora: 'text_clean' ahora incluye Título + Abstract + Keywords para máxima precisión.
"""

import re, regex
import unicodedata
import pandas as pd
from unidecode import unidecode
from pathlib import Path
import spacy
import logging
import sys

# ================= CONFIGURACIÓN =================
INPUT   = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusbloque1replacelematizar.csv"
# Generamos el nombre de salida automáticamente basado en el input
OUTPUT_CSV = str(Path(INPUT).parent / "datawos_scopusbloque1_cleanfinal2.csv")

COL_TITLE = "Title"
COL_ABS   = "Abstract"
COL_IDXKW = "Index Keywords"
COL_AUTHK = "Author Keywords"
SPACY_MODEL = "en_core_web_sm"

# ================= LISTAS DE PROTECCIÓN (DERECHO/GOBERNANZA) =================
# Esta lista protege n-gramas para que Spacy los trate como UN SOLO concepto.
# Alineada con RQs: 1 (Principios), 3 (Gobernanza/Brechas) y 4 (Temas).

EXCEPTION_PHRASES = [
    # --- 1. Actores y Geopolítica ---
    "china", "united states", "brazil", "canada", "india", "australia", "european union",
    "global south", "global north", "small island developing states", "sids",

    # --- 2. Organismos y Régimen Climático ---
    "united nations", "unfccc", "ipcc", "ipbes", "unep", "world polity",
    "international court of justice", "human rights council",
    
    # --- 3. Tratados e Instrumentos (Hard & Soft Law) ---
    "paris agreement", "kyoto protocol", "montreal protocol", "agenda 2030",
    "escazu agreement", "aarhus convention", "warsaw mechanism",
    
    # --- 4. Derecho y Principios (Responde a RQ1) ---
    "international environmental law", "environmental law", "administrative law",
    "precautionary principle", "preventive principle", 
    "polluter pays principle", "no harm rule",
    "common but differentiated responsibilities", "cbdr", # Clave para Equidad
    "intergenerational equity", "sustainable development",
    "common concern of humankind", "public trust doctrine",
    "duty to cooperate", "international cooperation", # Clave para Cooperación (RQ1)
    
    # --- 5. Gobernanza y Tensiones Institucionales (Responde a RQ3) ---
    "global environmental governance", "multilevel governance", "adaptive governance", 
    "polycentric governance", "climate governance", "transnational governance",
    "regime complex", "institutional fragmentation", "normative fragmentation", # Clave RQ3
    "compliance mechanisms", "dispute settlement",
    
    # --- 6. Justicia y Nuevas Tendencias (Responde a RQ4) ---
    "climate change", "global warming", 
    "climate justice", "procedural justice", "distributive justice",
    "climate litigation", "rights of nature", "human rights",
    "climate refugees", "climate migration", "loss and damage",
    "just transition", "net zero", "decarbonization",
    "geoengineering", "solar radiation management",
    "corporate social responsibility", "esg"
]

STOP_EXTRA = {
    "et","al","figure","fig","table","study","paper","using","use","based",
    "method","analysis","findings","introduction","discussion","data","research",
    "result","results","conclusion","author","review" # Agregué un par más comunes
}

STOP_EXTRA = {
    "et","al","figure","fig","table","study","paper","using","use","based",
    "method","analysis","findings","introduction","discussion","data","research"
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# ================= FUNCIONES =================
def basic_clean(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s)
    s = regex.sub(r"\s+", " ", s).strip()
    return s

def protect_exceptions(text: str) -> str:
    # Protege frases compuestas (climate change -> __EXC__climate_change__)
    for phrase in sorted(EXCEPTION_PHRASES, key=len, reverse=True):
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        token = f"__EXC__{phrase.lower().replace(' ', '_').replace('-', '_')}__"
        text = pattern.sub(token, text)
    return text

def restore_exceptions(text: str) -> str:
    return re.sub(r"__EXC__([a-z0-9_]+)__", lambda m: m.group(1).replace("_", " "), text)

def normalize_text_pipeline(s: str) -> str:
    s = basic_clean(s)
    s = protect_exceptions(s)
    s = unidecode(s.lower())
    s = basic_clean(s)
    return s

def normalize_keywords(val: str):
    if not val: return []
    parts = re.split(r";|\|", str(val))
    return [basic_clean(p).strip() for p in parts if basic_clean(p)]

# ================= MAIN =================
def main():
    logging.info(f"📂 Leyendo: {INPUT}")
    df = pd.read_csv(INPUT, encoding="utf-8", dtype=str).fillna("")

    # --- MODIFICACIÓN CLAVE AQUÍ ---
    # 1. Crear Texto Raw (Título + Abstract + Keywords)
    logging.info("🛠️  Construyendo texto base (Title + Abstract + Keywords)...")
    
    # Asegurar que no haya nulos
    df[COL_TITLE] = df[COL_TITLE].fillna("")
    df[COL_ABS]   = df[COL_ABS].fillna("")
    df[COL_IDXKW] = df[COL_IDXKW].fillna("")
    df[COL_AUTHK] = df[COL_AUTHK].fillna("")

    # Concatenamos TODO para que la IA tenga la máxima información posible
    df["text_raw"] = (
        df[COL_TITLE] + ". " + 
        df[COL_ABS] + ". " + 
        df[COL_IDXKW] + " " + 
        df[COL_AUTHK]
    ).apply(basic_clean)
    
    # Filtro de filas vacías (Si después de juntar todo sigue vacío, no sirve)
    initial_len = len(df)
    df = df[df["text_raw"].str.len() > 10].copy()
    logging.info(f"📉 Filas descartadas por estar vacías: {initial_len - len(df)}")

    # 2. Normalizar
    logging.info("🛡️  Normalizando...")
    df["text_prep"] = df["text_raw"].apply(normalize_text_pipeline)

    # 3. Lematizar con Spacy
    logging.info("🧠 Lematizando con Spacy...")
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    except:
        logging.error(f"Falta el modelo {SPACY_MODEL}. Ejecuta: python -m spacy download {SPACY_MODEL}")
        return

    stop_words = nlp.Defaults.stop_words.union(STOP_EXTRA) - {"no", "not"}

    clean_texts = []
    # Procesamos por lotes para velocidad
    for doc in nlp.pipe(df["text_prep"], batch_size=100):
        tokens = [t.lemma_ for t in doc if not (t.is_stop or t.is_punct or t.is_digit or t.lemma_ in stop_words)]
        clean_texts.append(" ".join(tokens))

    # Aquí se guarda la versión final que leerá el Script 2
    df["text_clean"] = [restore_exceptions(t) for t in clean_texts]

    # 4. Unificar Keywords (Para visualización humana en el Excel)
    logging.info("🔗 Unificando columna de keywords para reporte...")
    df["Keywords Unified"] = df.apply(
        lambda x: "; ".join(list(dict.fromkeys(normalize_keywords(x[COL_IDXKW]) + normalize_keywords(x[COL_AUTHK])))), 
        axis=1
    )

    # Limpieza final de columnas temporales
    df.drop(columns=["text_prep"], inplace=True, errors="ignore")
    
    logging.info(f"💾 Guardando: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logging.info("✅ Script 1 Terminado con éxito.")

if __name__ == "__main__":
    main()