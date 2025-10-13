# -*- coding: utf-8 -*-
import re
import pandas as pd
from pathlib import Path

# ========= CONFIG =========
IN_PATH  = r"G:/Mi unidad/Artículos cientificos/articulo 1/afiliaciones_detectadas_NOALIASES.csv"
OUT_PATH = str(Path(IN_PATH).with_name("_unique_organizationsfinal2final3.xlsx"))

# ========= CARGA =========
df = pd.read_csv(IN_PATH, dtype=str, keep_default_na=False)

# Verificación: solo usamos columnas YA normalizadas
for col in ["Affiliations_final", "Authors with affiliations_final"]:
    if col not in df.columns:
        raise ValueError(f"❌ Falta la columna requerida: {col}")

# ========= EXTRACCIÓN DE ORGANIZACIÓN =========
PRIORITY_PATTERNS = [
    re.compile(r"\buniv\w*", re.IGNORECASE),                # university / universidad
    re.compile(r"\binst\w*", re.IGNORECASE),                # institute / instituto
    re.compile(r"\bescuel\w*|\bschool\w*", re.IGNORECASE),  # escuela / school
    re.compile(r"\bcollege\b", re.IGNORECASE),
    re.compile(r"\bacad(?:emia\w*|emy\w*)", re.IGNORECASE),
    re.compile(r"\bfac(?:ultad\w*|ulty\w*)", re.IGNORECASE),
    re.compile(r"\bmuse(?:o|um)\w*", re.IGNORECASE),
    re.compile(r"\bhosp\w*", re.IGNORECASE),
    re.compile(r"\bcent(?:er|re|ro|rum)?\w*", re.IGNORECASE),
    re.compile(r"\bclin\w*", re.IGNORECASE),
    re.compile(r"\bminist(?:erio\w*|ry\w*)", re.IGNORECASE),
    re.compile(r"\blab\w*", re.IGNORECASE),
    re.compile(r"\bobserv\w*", re.IGNORECASE),
    re.compile(r"\bfund(?:acion\w*|aci[oó]n\w*|ation\w*)", re.IGNORECASE),
    re.compile(r"\bcorp\w*", re.IGNORECASE),
    re.compile(r"\bgov\w*", re.IGNORECASE),
    re.compile(r"\bauth\w*", re.IGNORECASE),
    re.compile(r"\bcons\w*", re.IGNORECASE),
    re.compile(r"\bserv\w*", re.IGNORECASE),
    re.compile(r"\b(?:depart\w*|dept\b|dep\b)", re.IGNORECASE),
    re.compile(r"\binvestig\w*", re.IGNORECASE),
]

def pick_primary_org(fragment: str):
    """Devuelve el segmento más institucional del fragmento (o el primero, si no hay match)."""
    if not fragment:
        return None
    segments = [s.strip(" ,;") for s in str(fragment).split(",") if s.strip(" ,;")]
    if not segments:
        return None
    for rx in PRIORITY_PATTERNS:
        for seg in segments:
            if rx.search(seg):
                return seg
    return segments[0]

def split_fragments(text: str):
    """Usa ';' como separador (si no hay ';', retorna el texto como un único fragmento)."""
    if not text:
        return []
    return [frag.strip(" ;") for frag in str(text).split(";")] if ";" in text else [text.strip()]

# ========= RECOPILAR Y DEDUP =========
candidatas = []
for col in ["Combined_universities_final"]:
    for frags in df[col].apply(split_fragments):
        for frag in frags:
            org = pick_primary_org(frag)
            if org:
                org = re.sub(r"\s+", " ", org).strip(" ,;.")
                if len(org) >= 2:
                    candidatas.append(org)

# De-dup preservando orden (case-insensitive)
unique_dict = {}
for org in candidatas:
    key = org.casefold()
    if key not in unique_dict:
        unique_dict[key] = org

unique_orgs = list(unique_dict.values())

# ========= EXPORTAR =========
out_df = pd.DataFrame({"Organization": unique_orgs})
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    out_df.to_excel(writer, index=False, sheet_name="unique_orgs")

print(f"✅ Organizaciones únicas: {len(unique_orgs):,}")
print(f"📄 Archivo guardado en: {OUT_PATH}")
