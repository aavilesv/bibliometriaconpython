# -*- coding: utf-8 -*-
"""
SCRIPT 02 FINAL: CLASIFICADOR + RESCATE ESTADÍSTICO + AUDITORÍA
---------------------------------------------------------------
1. Clasifica (70% IA / 30% Diccionario).
2. Limpia columnas de Citas y Años.
3. Aplica Rescate Automático (Top 10% Impacto + Top 15% Novedad).
4. Genera Muestra de Auditoría (10% Aleatorio) para tu validación del 96%.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

# ================= CONFIGURACIÓN =================
# Tus rutas originales
INPUT_FILE = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusbloque1_cleanfinal2.csv"
OUTPUT_DIR = Path(r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\outputs_classifier")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

COL_TEXT_CLEAN = "text_clean" 
COL_ABS_ORIG   = "Abstract"

# Pesos (70% IA / 30% Diccionario) - ESTRATEGIA CONSERVADORA
W_EMBED = 0.70
W_DICT  = 0.30

# ================= DEFINICIÓN DEL TEMA (Tus Listas Correctas) =================
REFERENCE_TEXTS = [
    "Polycentric and multilevel governance systems addressing climate change adaptation through transnational networks and non-state actors.",
    "The reconfiguration of international environmental law principles, specifically equity, precaution, and common but differentiated responsibilities.",
    "The intersection of human rights litigation, climate justice, and intergenerational equity in international courts.",
    "Epistemic communities and the influence of scientific evidence on global environmental policy-making and treaty design.",
    "Challenges in regulatory efficacy, compliance mechanisms, and the normative impact of the Paris Agreement on state obligations."
]

POSITIVE_TERMS = [
    r"\binternational (environmental )?law\b", r"\bglobal governance\b", r"\bclimate governance\b",
    r"\bparis agreement\b", r"\bunfccc\b", r"\bkyoto protocol\b", r"\btreat(y|ies)\b",
    r"\bhuman rights\b", r"\bclimate justice\b", r"\blitigation\b", r"\bcourts?\b", 
    r"\brights of nature\b", r"\bprocedural justice\b", r"\bdistributive justice\b",
    r"\bclimate refugees?\b", r"\bjust transition\b",
    r"\bsoft law\b", r"\bhard law\b", r"\blegal framework\b", r"\bregulatory\b", 
    r"\bnormative\b", r"\bfragmentation\b", r"\bcompliance\b",
    r"\bprecautionary principle\b", r"\bpolluter pays\b",
    r"\bcommon but differentiated\b", r"\bcbdr\b",
    r"\bintergenerational equity\b", r"\bsustainable development\b",
    r"\bno harm rule\b", r"\bduty to cooperate\b",
    r"\bpolycentric\b", r"\bmultilevel\b", r"\badaptive governance\b", 
    r"\bepistemic\b", r"\btechno[\-\s]?scientific\b", r"\bnon[\-\s]state actors\b",
    r"\bregime complex\b", r"\bloss and damage\b", r"\bndcs?\b",
    r"\bnet zero\b", r"\bdecarbonization\b", r"\bgeoengineering\b"
]

NEGATIVE_TERMS = [
    r"\bchemical engineering\b", r"\bmolecular biology\b", r"\bclinical trial\b",
    r"\bpatient\b", r"\bsurgery\b", r"\bpolymer\b", r"\balgorithm optimization\b",
    r"\bwireless sensor network\b", r"\bcrop yield\b", r"\bsoil chemistry\b",
    r"\brenewable energy storage\b", r"\bphotovoltaic\b", r"\bbiomass production\b"
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
    ref_emb = model.encode(REFERENCE_TEXTS, convert_to_tensor=True)
    art_emb = model.encode(df_texts.tolist(), convert_to_tensor=True, show_progress_bar=True)
    scores = util.cos_sim(art_emb, ref_emb).mean(axis=1).cpu().numpy()
    return (scores - scores.min()) / (scores.max() - scores.min())

def get_dictionary_score(text):
    if not isinstance(text, str): return 0
    text = text.lower()
    pos = sum(1 for p in POSITIVE_TERMS if re.search(p, text))
    neg = sum(1 for p in NEGATIVE_TERMS if re.search(p, text))
    return 1 - np.exp(-0.5 * max(0, pos - (1.5 * neg)))

def classify_article(score):
    if score >= 0.75: return "🔥 ALTA RELEVANCIA"
    if score >= 0.60: return "✅ MEDIA RELEVANCIA" # (Rango ampliado para rescate)
    if score >= 0.45: return "⚠️ BAJA RELEVANCIA"
    return "❌ DESCARTAR"

# ================= PROCESO PRINCIPAL =================
def main():
    print(f"📂 Leyendo: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False, dtype=str)
    
    # 1. FILTRADO ROBUSTO
    df = df[df[COL_ABS_ORIG].notna() & (df[COL_ABS_ORIG].str.len() > 50) & df[COL_TEXT_CLEAN].notna()].copy()
    
    # 2. CÁLCULO DE SCORES
    print("🧠 Calculando Scores...")
    df['score_semantic'] = get_embedding_score(df[COL_TEXT_CLEAN])
    df['score_dict'] = df[COL_TEXT_CLEAN].apply(get_dictionary_score)
    df['FINAL_SCORE'] = (df['score_semantic'] * W_EMBED) + (df['score_dict'] * W_DICT)
    df['DECISION'] = df['FINAL_SCORE'].apply(classify_article)

    # 3. LIMPIEZA DE METADATOS (CRUCIAL PARA EL RESCATE)
    print("⚙️ Normalizando Citas y Años...")
    if "Cited by" in df.columns:
        df["Cited by"] = df["Cited by"].fillna("0").astype(str).str.replace(r"[^\d]", "", regex=True).replace("", "0").astype(int)
    else:
        df["Cited by"] = 0
    
    if "Year" in df.columns:
        df["Year"] = df["Year"].fillna("0").astype(str).str.replace(r"[^\d]", "", regex=True).replace("", "0").astype(int)

    # 4. === PROTOCOLO DE RESCATE AUTOMÁTICO (PERCENTILES) ===
    print("🤖 Ejecutando Rescate Estadístico...")
    
    # Grupo A: ALTA (Pasan todos)
    df_high = df[df['DECISION'] == "🔥 ALTA RELEVANCIA"].copy()
    df_high['Rescate_Reason'] = "Directo: Alta Relevancia"

    # Grupo B: MEDIA (De aquí rescatamos)
    df_mid = df[df['DECISION'] == "✅ MEDIA RELEVANCIA"].copy()
    total_mid = len(df_mid)
    
    # --- Criterio 1: Impacto (Top 10% más citados o mínimo 5) ---
    n_rescue_impact = max(5, int(total_mid * 0.10))
    rescue_impact = df_mid.sort_values(by="Cited by", ascending=False).head(n_rescue_impact).copy()
    rescue_impact['Rescate_Reason'] = f"Rescate: Impacto (Top {n_rescue_impact} citados)"
    
    # --- Criterio 2: Novedad (Top 15% de los recientes 2023-25) ---
    ids_impact = rescue_impact.index.tolist()
    # Filtramos recientes que NO estén ya seleccionados
    df_mid_recent = df_mid[(df_mid["Year"] >= 2023) & (~df_mid.index.isin(ids_impact))].copy()
    
    if len(df_mid_recent) > 0:
        n_rescue_novelty = max(3, int(len(df_mid_recent) * 0.15))
        rescue_novelty = df_mid_recent.sort_values(by="FINAL_SCORE", ascending=False).head(n_rescue_novelty).copy()
        rescue_novelty['Rescate_Reason'] = f"Rescate: Novedad (Top {n_rescue_novelty} recientes)"
    else:
        rescue_novelty = pd.DataFrame(columns=df_mid.columns)

    # UNIFICACIÓN
    df_final_sample = pd.concat([df_high, rescue_impact, rescue_novelty])
    df_final_sample = df_final_sample.sort_values(by=["DECISION", "FINAL_SCORE"], ascending=False)

    # 5. === GENERACIÓN DE AUDITORÍA (10% ALEATORIO) ===
    print("🎲 Generando Muestra de Auditoría...")
    df_excluded = df[~df.index.isin(df_final_sample.index)].copy()
    
    # Muestra del 10%, acotada entre 20 y 50 artículos
    audit_size = int(len(df_excluded) * 0.10)
    audit_size = max(20, min(50, audit_size))
    audit_size = min(audit_size, len(df_excluded)) 
    
    audit_sample = df_excluded.sample(n=audit_size, random_state=42)

    # REPORTE EN CONSOLA
    print(f"\n📊 RESUMEN FINAL:")
    print(f"   - Candidatos en Media: {total_mid}")
    print(f"   - Rescatados Impacto:  {len(rescue_impact)}")
    print(f"   - Rescatados Novedad:  {len(rescue_novelty)}")
    print(f"   ---------------------------------------")
    print(f"   TOTAL PARA LEER:       {len(df_final_sample)} artículos")
    print(f"   TOTAL PARA VALIDAR:    {len(audit_sample)} artículos (Auditoría)")

    # 6. EXPORTACIÓN
    excel_path = OUTPUT_DIR / "MUESTRA_FINAL_DEFINITIVA.xlsx"
    print("💾 Guardando Excel...")

    cols_export = ['Rescate_Reason', 'FINAL_SCORE', 'Cited by', 'Year', 'Title', 'Abstract', 'DOI', 'score_semantic', 'score_dict']
    cols_exist = [c for c in cols_export if c in df.columns]

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # HOJA 1: TU MUESTRA DE LECTURA (Aquí está todo listo)
        df_final_sample[cols_exist].to_excel(writer, sheet_name='1_LECTURA_OBLIGATORIA', index=False)
        
        # HOJA 2: TU MUESTRA DE AUDITORÍA (Para validar el 96%)
        audit_sample[['DECISION', 'FINAL_SCORE', 'Title', 'Abstract']].to_excel(writer, sheet_name='2_AUDITORIA_RAPIDA', index=False)
        
        # HOJA 3: Backup Estadísticas
        df['DECISION'].value_counts().to_frame("Total").to_excel(writer, sheet_name='Stats')

    print(f"✅ ¡LISTO! Archivo generado: {excel_path}")

if __name__ == "__main__":
    main()