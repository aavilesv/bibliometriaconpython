# -*- coding: utf-8 -*-
"""
MERGE SCORES INTO PRINCIPAL (TÍTULO primero, luego DOI) — CSV
Autor: Angelo Avilés + IA Assistant
Fecha: 2025-10-16
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

# ===================== RUTAS =====================
PRINCIPAL_CSV = Path(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replacelematizar.csv")
FILTRO_PATH   = Path(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\outputs_classifier\article_prioritization1.xlsx")

OUT_DIR       = Path(r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\outputs_classifier")
OUT_CSV_ALL   = OUT_DIR / "article_referencias_all_scores.csv"
OUT_CSV_FIL   = OUT_DIR / "article_referencias_filtered_bucket.csv"  # Borderline + Recommended

print(f"PRINCIPAL: {PRINCIPAL_CSV}  -> existe: {PRINCIPAL_CSV.exists()}")
print(f"FILTRO   : {FILTRO_PATH}    -> existe: {FILTRO_PATH.exists()}")

# ===================== CONFIG =====================
PRINCIPAL_DOI_COL   = "DOI"
PRINCIPAL_TITLE_COL = "Title"
SCORE_COLS = ["embed_score", "dict_score", "relevance_score", "supervised_prob", "bucket"]

# ===================== HELPERS =====================
def norm_doi(x) -> str:
    if pd.isna(x): return ""
    x = str(x).strip()
    x = re.sub(r"https?://(dx\.)?doi\.org/", "", x, flags=re.I)
    return x.lower().strip()

def norm_title(x) -> str:
    if pd.isna(x): return ""
    x = str(x).lower().strip()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def to_num(s):
    """Convierte a num para ordenar 'mejor fila'; NaN si no es convertible."""
    try:
        return float(s)
    except Exception:
        return np.nan

# ===================== CARGA =====================
principal = pd.read_csv(PRINCIPAL_CSV, low_memory=False, dtype=str).fillna("")

# Leer Excel del filtro (hoja All_scores si existe)
try:
    filtro_raw = pd.read_excel(FILTRO_PATH, sheet_name="All_scores", dtype=str).fillna("")
except ValueError:
    filtro_raw = pd.read_excel(FILTRO_PATH, dtype=str).fillna("")

# Normalizar nombres posibles del filtro
colmap = {c.lower(): c for c in filtro_raw.columns}
need = {
    "title": None, "doi": None, "year": None, "abstract": None,
    "embed_score": None, "dict_score": None, "relevance_score": None,
    "supervised_prob": None, "bucket": None,
}
for k in list(need.keys()):
    if k in colmap:
        need[k] = colmap[k]
    elif k == "year" and "Year" in filtro_raw.columns:
        need["year"] = "Year"

missing = [k for k, v in need.items() if v is None]
if missing:
    raise ValueError(f"En el FILTRO faltan columnas requeridas: {missing}. "
                     f"Columnas disponibles: {list(filtro_raw.columns)}")

# Nos quedamos con lo necesario del filtro
filtro = filtro_raw[[need["title"], need["doi"], need["year"], need["abstract"]] +
                    [need[c] for c in ["embed_score","dict_score","relevance_score","supervised_prob","bucket"]]].copy()
filtro.columns = ["Title","DOI","year","Abstract"] + SCORE_COLS

# ===================== VALIDACIÓN MÍNIMA =====================
for c in [PRINCIPAL_DOI_COL, PRINCIPAL_TITLE_COL]:
    if c not in principal.columns:
        raise ValueError(f"Falta la columna '{c}' en el PRINCIPAL.")

# ===================== NORMALIZACIÓN DE CLAVES =====================
principal["_title_norm"] = principal[PRINCIPAL_TITLE_COL].map(norm_title)
principal["_doi_norm"]   = principal[PRINCIPAL_DOI_COL].map(norm_doi)

filtro["_title_norm"]    = filtro["Title"].map(norm_title)
filtro["_doi_norm"]      = filtro["DOI"].map(norm_doi)

# ===================== ELEGIR LA "MEJOR FILA" POR CLAVE =====================
# Criterio: mayor relevance_score, luego mayor supervised_prob (si existen)
filtro["_relevance_num"] = filtro["relevance_score"].map(to_num)
filtro["_superv_num"]    = filtro["supervised_prob"].map(to_num)

# Para TÍTULO (prioritario): usar solo filas con título normalizado no vacío
f_titulo = (filtro[filtro["_title_norm"] != ""]
            .sort_values(["_relevance_num","_superv_num"], ascending=[False, False])
            .drop_duplicates(subset=["_title_norm"], keep="first")
            .drop(columns=["_relevance_num","_superv_num"]))

# Para DOI (solo cuando no hay título): usar filas cuyo título normalizado esté vacío pero DOI exista
f_doi_sin_titulo = (filtro[(filtro["_title_norm"] == "") & (filtro["_doi_norm"] != "")]
                    .sort_values(["_relevance_num","_superv_num"], ascending=[False, False])
                    .drop_duplicates(subset=["_doi_norm"], keep="first")
                    .drop(columns=["_relevance_num","_superv_num"]))

# Nos quedamos solo con columnas de score para el merge
f_titulo_min = f_titulo[["_title_norm"] + SCORE_COLS]
f_doi_min    = f_doi_sin_titulo[["_doi_norm"] + SCORE_COLS]

# ===================== MERGE 1: POR TÍTULO (PRIORIDAD) =====================
m = principal.merge(f_titulo_min, on="_title_norm", how="left")

# ===================== MERGE 2: POR DOI (SOLO FILAS SIN TÍTULO O SIN SCORE) =====================
# criterio: SIN título normalizado (vacío) o no llenó aún los scores
faltan_scores = m["embed_score"].isna()
sin_titulo    = (m["_title_norm"] == "")
pendientes    = m[faltan_scores & sin_titulo].copy()

if not pendientes.empty:
    m2 = pendientes.drop(columns=SCORE_COLS).merge(f_doi_min, on="_doi_norm", how="left")
    m.update(m2)  # solo actualiza esas filas

# ===================== DEDUPLICACIÓN (sin inflar filas) =====================
# Clave preferente: título normalizado; si está vacío, usar DOI normalizado.
m = m.reset_index(drop=False).rename(columns={"index": "_row_index"})
m["_dedup_key"] = m.apply(
    lambda r: ("t:" + r["_title_norm"]) if r["_title_norm"] 
              else (("d:" + r["_doi_norm"]) if r["_doi_norm"] else ("i:" + str(r["_row_index"]))),
    axis=1
)

before = len(m)
m = m.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
after  = len(m)
print(f"Deduplicación: eliminadas {before - after} filas; quedan {after}.")

# ===================== LIMPIEZA DE AUXILIARES =====================
aux_cols = ["_title_norm","_doi_norm","_dedup_key","_row_index"]
m_final = m.drop(columns=[c for c in aux_cols if c in m.columns])

# ===================== EXPORTACIÓN =====================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CSV completo (alineado 1:1 con principal, sin columnas extra)
m_final.to_csv(OUT_CSV_ALL, index=False, encoding="utf-8-sig")

# CSV filtrado por bucket
bucket_keep = {"Borderline", "Recommended"}
m_filtrado = m_final[m_final["bucket"].astype(str).isin(bucket_keep)].copy()
m_filtrado.to_csv(OUT_CSV_FIL, index=False, encoding="utf-8-sig")

# ===================== RESUMEN =====================
print("✅ Proceso terminado.")
print(f"Filas en CSV completo: {len(m_final)}  -> {OUT_CSV_ALL}")
print(f"Filas en CSV filtrado (bucket Borderline/Recommended): {len(m_filtrado)}  -> {OUT_CSV_FIL}")
