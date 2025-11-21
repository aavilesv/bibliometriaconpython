# -*- coding: utf-8 -*-
"""
SCRIPT 02: CLASIFICADOR PRO (NIVELES DE PRIORIDAD)
--------------------------------------------------
- Elimina filas sin Abstract (Robustez garantizada).
- Clasifica en: Alta (0.8+), Media (0.65+), Baja (0.5+).
- Incluye todas las columnas bibliométricas solicitadas.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

# ================= CONFIGURACIÓN =================
INPUT_FILE = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusbloque1_cleanfinal.csv"
OUTPUT_DIR = Path(r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\outputs_classifier")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Columnas Clave
COL_TEXT_CLEAN = "text_clean" 
COL_ABS_ORIG   = "Abstract"  # Para verificar que no esté vacío

# Pesos (80% IA / 20% Diccionario)
W_EMBED = 0.80
W_DICT  = 0.20

# ================= DEFINICIÓN DEL TEMA (Gobernanza y Derecho) =================
REFERENCE_TEXTS = [
    "Polycentric and multilevel governance systems addressing climate change adaptation through transnational networks.",
    "The fragmentation of international environmental law and the evolution of soft law into hard law frameworks.",
    "The intersection of human rights litigation and climate justice in international courts.",
    "Epistemic communities and the influence of technoscientific evidence on global environmental policy-making.",
    "Challenges in regulatory efficacy, compliance mechanisms, and the normative impact of the Paris Agreement."
]

POSITIVE_TERMS = [
    r"\binternational (environmental )?law\b", r"\bglobal governance\b", r"\bclimate governance\b",
    r"\bparis agreement\b", r"\bunfccc\b", r"\bkyoto protocol\b", r"\btreat(y|ies)\b",
    r"\bhuman rights\b", r"\bclimate justice\b", r"\blitigation\b", r"\bcourts?\b", 
    r"\brights of nature\b", r"\bprocedural justice\b",
    r"\bsoft law\b", r"\bhard law\b", r"\blegal framework\b", r"\bregulatory\b", 
    r"\bnormative\b", r"\bfragmentation\b", r"\bcompliance\b",
    r"\bpolycentric\b", r"\bmultilevel\b", r"\badaptive governance\b", 
    r"\bepistemic\b", r"\btechno[\-\s]?scientific\b", r"\bnon[\-\s]state actors\b"
]

NEGATIVE_TERMS = [
    r"\bchemical engineering\b", r"\bmolecular biology\b", r"\bclinical trial\b",
    r"\bpatient\b", r"\bsurgery\b", r"\bpolymer\b", r"\balgorithm optimization\b",
    r"\bwireless sensor network\b", r"\bcrop yield\b", r"\bsoil chemistry\b"
]

# ================= CARGA DE MODELO =================
print("⏳ Cargando Modelo MPNet...")
try:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-mpnet-base-v2")
    print("✅ Modelo cargado.")
except ImportError:
    print("❌ ERROR: Falta 'sentence-transformers'.")
    sys.exit()

# ================= FUNCIONES =================

def get_embedding_score(df_texts):
    # Codificar y Comparar
    ref_emb = model.encode(REFERENCE_TEXTS, convert_to_tensor=True)
    art_emb = model.encode(df_texts.tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    # Usamos la media de similitud contra las 5 referencias
    scores = util.cos_sim(art_emb, ref_emb).mean(axis=1).cpu().numpy()
    
    # Normalizar (Min-Max Scaling) para tener valores entre 0 y 1 reales
    return (scores - scores.min()) / (scores.max() - scores.min())

def get_dictionary_score(text):
    if not isinstance(text, str): return 0
    text = text.lower()
    pos = sum(1 for p in POSITIVE_TERMS if re.search(p, text))
    neg = sum(1 for p in NEGATIVE_TERMS if re.search(p, text))
    return 1 - np.exp(-0.5 * max(0, pos - (1.5 * neg)))

def classify_article(score):
    """Tu sistema de clasificación solicitado"""
    if score >= 0.80: return "🔥 ALTA RELEVANCIA"
    if score >= 0.65: return "✅ MEDIA RELEVANCIA"
    if score >= 0.50: return "⚠️ BAJA RELEVANCIA"
    return "❌ DESCARTAR"

# ================= PROCESO PRINCIPAL =================
def main():
    print(f"📂 Leyendo: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False, dtype=str)
    
    # 1. VALIDACIÓN DE ROBUSTEZ (Eliminar filas sin Abstract)
    print("🧹 Filtrando documentos vacíos o incompletos...")
    initial_len = len(df)
    
    # Aseguramos que tenga Abstract y que sea mayor a 50 caracteres (para evitar "No abstract available")
    df = df[
        df[COL_ABS_ORIG].notna() & 
        (df[COL_ABS_ORIG].str.len() > 50) &
        df[COL_TEXT_CLEAN].notna()
    ].copy()
    
    filtered_len = len(df)
    print(f"   -> Descartados {initial_len - filtered_len} artículos por falta de Abstract.")
    print(f"   -> Procesando {filtered_len} artículos robustos.")

    if filtered_len == 0:
        print("❌ Error: No quedan artículos después del filtrado. Revisa tu CSV.")
        return

    # 2. ANÁLISIS
    print("🧠 Analizando similitud semántica...")
    df['score_semantic'] = get_embedding_score(df[COL_TEXT_CLEAN])
    
    print("📖 Analizando diccionario...")
    df['score_dict'] = df[COL_TEXT_CLEAN].apply(get_dictionary_score)
    
    # 3. SCORE FINAL
    df['FINAL_SCORE'] = (df['score_semantic'] * W_EMBED) + (df['score_dict'] * W_DICT)
    df['DECISION'] = df['FINAL_SCORE'].apply(classify_article)

    # 4. SELECCIÓN DE COLUMNAS (Las que pediste)
    # Verificamos que existan en el CSV, si no, las creamos vacías para que no falle
    requested_cols = [
        "Source title", "Author full names", "Affiliations", 
        "Document Type", "Cited by", "Keywords Unified"
    ]
    for c in requested_cols:
        if c not in df.columns:
            df[c] = "" # Rellenar vacíos si falta alguna

    # Columnas para el reporte ordenadas
    report_cols = [
        'DECISION', 'FINAL_SCORE', 'score_semantic', 'score_dict',
        'Title', 'Year', 'DOI', 'Abstract'
    ] + requested_cols

    # Ordenar por relevancia
    df_sorted = df.sort_values(by='FINAL_SCORE', ascending=False)

    # 5. EXPORTACIÓN
    excel_path = OUTPUT_DIR / "SELECCION_FINAL_ROBUSTAfinal.xlsx"
    print("💾 Generando Excel ordenado...")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Hoja 1: Resumen Estadístico
        summary = df['DECISION'].value_counts().to_frame("Total Artículos")
        summary.to_excel(writer, sheet_name='Resumen')

        # Hoja 2: ALTA RELEVANCIA (0.8 - 1.0) -> Lo mejor de lo mejor
        high = df_sorted[df_sorted['DECISION'] == "🔥 ALTA RELEVANCIA"]
        high[report_cols].to_excel(writer, sheet_name='1_ALTA_PRIORIDAD', index=False)
        
        # Hoja 3: MEDIA RELEVANCIA (0.65 - 0.79) -> Muy buenos
        mid = df_sorted[df_sorted['DECISION'] == "✅ MEDIA RELEVANCIA"]
        mid[report_cols].to_excel(writer, sheet_name='2_MEDIA_PRIORIDAD', index=False)

        # Hoja 4: BAJA RELEVANCIA (0.50 - 0.64) -> Revisar si faltan
        low = df_sorted[df_sorted['DECISION'] == "⚠️ BAJA RELEVANCIA"]
        low[report_cols].to_excel(writer, sheet_name='3_BAJA_PRIORIDAD', index=False)

        # Hoja 5: TODO (Backup)
        df_sorted[report_cols].to_excel(writer, sheet_name='Todos_los_Datos', index=False)

    print(f"✅ ¡Proceso Completado! Archivo: {excel_path}")
    print(f"   🔥 Alta Prioridad: {len(high)}")
    print(f"   ✅ Media Prioridad: {len(mid)}")
    print(f"   ⚠️ Baja Prioridad: {len(low)}")

if __name__ == "__main__":
    main()