# =========================
# Limpieza y fusión de afiliaciones (WoS/Scopus)
# - Prioriza nombre de organización (Universidad/Instituto/etc.)
# - Adjunta país detectado en el fragmento
# - Deduplicación difusa (RapidFuzz)
# =========================

import pandas as pd                          # Manejo de dataframes
import re                                    # Expresiones regulares
from rapidfuzz import fuzz, process          # Fuzzy matching para dedupe
from typing import Optional                  # Tipado para funciones

# ---------- 1) Rutas de E/S ----------
ruta = r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replacelematizar.csv"  # CSV de entrada
OUT  = r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusafiliation.csv"               # CSV de salida

# ---------- 2) Cargar CSV (forzando string) ----------
df = pd.read_csv(ruta, low_memory=False, dtype=str).fillna("")  # Evita DtypeWarning; convierte NaN a cadena vacía

# ---------- 3) Lista de países (canónica) ----------
countries = [
    "Algeria", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Canada", "China",
    "Colombia", "Costa Rica", "Czech Republic", "Denmark", "Egypt", "Chile",
    "Ethiopia", "Finland", "France", "Germany", "Ghana",
    "Greece", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Lebanon",
    "Lithuania", "Malaysia", "Mexico", "Morocco", "Nepal", "Netherlands",
    "New Zealand", "Nigeria", "Norway", "Oman", "Pakistan", "Palestine",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
    "Russia", "Rwanda", "Saudi Arabia", "Senegal", "Serbia", "Singapore",
    "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka",
    "Sweden", "Switzerland", "Taiwan", "Thailand", "Tunisia", "Turkey",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Uzbekistan", "Vietnam", "Zimbabwe", "Ivory Coast", "Kuwait",
    "Croatia", "Afghanistan", "Albania", "Andorra", "Angola", "Armenia",
    "Azerbaijan", "Bahamas", "Bahrain", "Barbados", "Belarus", "Belize",
    "Benin", "Bhutan", "Bolivia", "Botswana", "Brunei Darussalam",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon",
    "Central African Republic", "Chad", "Comoros", "Congo", "Cuba",
    "Cyprus", "Djibouti", "Dominican Republic", "Ecuador", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Fiji", "Gabon",
    "Gambia", "Georgia", "Grenada", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Iceland", "Jamaica", "Kiribati", "Kyrgyzstan", "Laos",
    "Latvia", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Luxembourg",
    "Madagascar", "Malawi", "Maldives", "Mali", "Malta", "Marshall Islands",
    "Mauritania", "Mauritius", "Micronesia", "Monaco", "Mongolia",
    "Montenegro", "Mozambique", "Myanmar", "Namibia", "Nauru", "Niger",
    "North Macedonia", "Palau", "Panama", "Papua New Guinea", "Paraguay",
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Seychelles", "Sierra Leone", "Slovakia",
    "Solomon Islands", "Somalia", "South Sudan", "Sudan", "Suriname",
    "Tajikistan", "Tanzania", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Turkmenistan", "Tuvalu", "Uganda", "Uruguay",
    "Vanuatu", "Venezuela", "Yemen", "Zambia", "Swaziland"
]

# ---------- 4) Diccionario de sinónimos (minúscula->canónico) ----------
synonyms_map = {c.lower(): c for c in countries}  # Permite normalizar "china" -> "China"

# ---------- 5) Compilar patrones de normalización ----------
country_vals = '|'.join(map(re.escape, synonyms_map.values()))                                    # Países escapados para regex
sep_pattern = re.compile(rf'({country_vals})\s*,\s*(?=[A-Z])')                                    # Corrige "País, Siguiente" -> "País; Siguiente"
bracket_pattern = re.compile(r'\[.*?\]')                                                          # Elimina "[Autor]" (metadatos de WoS)

# ---------- 6) Regex para extraer país en cualquier parte del fragmento ----------
COUNTRY_REGEX = re.compile(r'\b(' + '|'.join(map(re.escape, synonyms_map.values())) + r')\b', re.IGNORECASE)

def extract_country(fragment: str) -> Optional[str]:
    """Devuelve el país canónico detectado dentro del fragmento (última coincidencia si hay varias)."""
    if not fragment:                                           # Fragmento vacío
        return None
    hits = COUNTRY_REGEX.findall(fragment)                     # Busca países por regex
    if not hits:                                               # Si no hay, retorna None
        return None
    canonical = synonyms_map.get(hits[-1].lower())             # Toma la última ocurrencia (suele ir al final)
    return canonical or hits[-1]                               # Devuelve en forma canónica

def normalize_cell(raw: str) -> str:
    """
    Limpia un campo de afiliaciones:
    - Elimina corchetes y su contenido (p. ej., [Autor])
    - Corrige separadores "País, Next" -> "País; Next"
    - Asegura separación por '; ' sin entradas vacías
    - Intenta propagar país al fragmento anterior cuando el patrón de país está al inicio del siguiente
    """
    text = bracket_pattern.sub('', raw)                        # Quita [..] (WoS)
    text = sep_pattern.sub(r'\1; ', text)                      # Reemplaza "País, Siguiente" por "País; Siguiente"
    frags = [frag.strip() for frag in text.split(';') if frag.strip()]  # Divide por ';' ya limpio

    norm = []                                                  # Lista normalizada de fragmentos
    for i, frag in enumerate(frags):                           # Itera fragmentos
        if any(frag.endswith(c) for c in synonyms_map.values()):   # Si ya termina en un país, lo deja
            norm.append(frag)
        else:
            if i + 1 < len(frags):                             # Mira el siguiente para detectar país inicial
                nxt = frags[i+1]
                # Si el siguiente empieza con un país, propaga al actual
                m = COUNTRY_REGEX.match(nxt)                   # match desde el inicio
                if m:
                    country = synonyms_map.get(m.group(1).lower(), m.group(1))
                    norm.append(f"{frag}, {country}")          # Anexa país detectado
                else:
                    norm.append(frag)                          # Si no hay país claro, deja tal cual
            else:
                norm.append(frag)                              # Último fragmento sin país: lo deja
    return '; '.join(norm)                                     # Une con '; ' para coherencia

# ---------- 7) Columnas a usar: intenta *_final y, si no existen, cae a originales ----------
COL_A = "Affiliations_final"                                   # Preferido si existe
COL_B = "Authors with affiliations_final"                      # Preferido si existe
if COL_A not in df.columns:                                    # Si no existe, usa "Affiliations"
    COL_A = "Affiliations"
if COL_B not in df.columns:                                    # Si no existe, usa "Authors with affiliations"
    COL_B = "Authors with affiliations"

# ---------- 8) Aplica normalización a ambas columnas de entrada ----------
df[COL_A] = df[COL_A].apply(normalize_cell)                    # Normaliza separadores/país
df[COL_B] = df[COL_B].apply(normalize_cell)                    # Ídem para autores+afil.

# ---------- 9) Umbral de deduplicación difusa ----------
DEDUPE_THRESHOLD = 98                                          # >=98 se considera duplicado (token_set_ratio)

# ---------- 10) Patrones de prioridad (qué se considera "organización principal") ----------
PRIORITY_PATTERNS = [
    re.compile(r"\buniv\w*", re.IGNORECASE),                   # Universidad / University
    re.compile(r"\binst\w*", re.IGNORECASE),                   # Instituto / Institute
    re.compile(r"\bescue\w*", re.IGNORECASE),                  # Escuela (forma truncada común)
    re.compile(r"^\s*school\w*", re.IGNORECASE),               # School
    re.compile(r"^\s*college\b", re.IGNORECASE),               # College
    re.compile(r"^\s*acad(?:emia\w*|emy\w*)", re.IGNORECASE),  # Academia / Academy
    re.compile(r"^\s*fac(?:ultad\w*|ulty\w*)", re.IGNORECASE), # Facultad / Faculty
    re.compile(r"^\s*muse(?:o|um)\w*", re.IGNORECASE),         # Museo / Museum
    re.compile(r"^\s*hosp\w*", re.IGNORECASE),                 # Hospital
    re.compile(r"^\s*cent(?:er|re|ro|rum)?\w*", re.IGNORECASE),# Center/Centre/Centro
    re.compile(r"^\s*clin\w*", re.IGNORECASE),                 # Clínica / Clinic
    re.compile(r"^\s*minist(?:erio\w*|ry\w*)", re.IGNORECASE), # Ministerio / Ministry
    re.compile(r"^\s*lab\w*", re.IGNORECASE),                  # Laboratorio / Lab
    re.compile(r"^\s*observ\w*", re.IGNORECASE),               # Observatorio / Observatory
    re.compile(r"^\s*fund(?:acion\w*|aci[oó]n\w*|ation\w*)", re.IGNORECASE), # Fundación
    re.compile(r"^\s*corp\w*", re.IGNORECASE),                 # Corporación / Corp
    re.compile(r"^\s*gov\w*", re.IGNORECASE),                  # Government
    re.compile(r"^\s*auth\w*", re.IGNORECASE),                 # Authority
    re.compile(r"^\s*cons\w*", re.IGNORECASE),                 # Consejo / Council / Consortium / Consulting
    re.compile(r"^\s*serv\w*", re.IGNORECASE),                 # Servicio / Service
    re.compile(r"^\s*(?:depart\w*|dept\b|dep\b)", re.IGNORECASE), # Department
    re.compile(r"^\s*flac\w*", re.IGNORECASE),                 # FLACSO, etc.
    re.compile(r"^\s*investig\w*", re.IGNORECASE),             # Investigación / Research
]

def pick_primary_org(fragment: str) -> Optional[str]:
    """
    Selecciona el nombre de la organización prioritaria DENTRO del fragmento,
    y le ANEXA el país detectado en el fragmento completo (si existe y no está incluido).
    """
    if not fragment or not str(fragment).strip():              # Fragmento vacío
        return None

    country = extract_country(fragment)                        # Detecta país a nivel de fragmento completo

    segments = [s.strip() for s in str(fragment).split(',') if s.strip()]  # Trocea por coma para buscar la org.
    if not segments:                                           # Sin segmentos útiles
        return None

    org_seg = None                                             # Candidato a organización principal
    for rx in PRIORITY_PATTERNS:                               # Recorre patrones de prioridad
        for seg in segments:                                   # Examina cada segmento
            if rx.search(seg):                                 # Si calza patrón (univ/inst/etc.)
                org_seg = seg                                  # Toma este segmento como organización
                break
        if org_seg:                                            # Si ya encontramos, rompe bucle superior
            break

    if not org_seg:                                            # Fallback si no calzaron patrones
        for seg in segments:
            if re.search(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]', seg):      # Toma el primer segmento "alfabético"
                org_seg = seg
                break

    if not org_seg:                                            # Si aún no hay nada, retorna None
        return None

    if country and not re.search(rf'\b{re.escape(country)}\b', org_seg, re.IGNORECASE):
        return f"{org_seg}, {country}"                         # Anexa ", País" si no está ya presente
    return org_seg                                             # Devuelve solo organización si ya incluye país

def dedupe_fuzzy(items: list[str], threshold: int) -> list[str]:
    """
    Deduplicación difusa (token_set_ratio) manteniendo la variante más larga.
    threshold >= 0 activa fuzzy; threshold <= 0 hace dedupe exacto insensible a mayúsculas.
    """
    out: list[str] = []                                        # Resultado
    if threshold <= 0:                                         # Dedupe exacto si threshold <= 0
        seen = set()
        for it in items:
            key = it.lower()
            if key not in seen:
                out.append(it); seen.add(key)
        return out

    for cand in items:                                         # Recorre candidatos
        if not out:                                            # Si lista vacía, agrega directo
            out.append(cand); continue
        best = process.extractOne(cand, out, scorer=fuzz.token_set_ratio)  # Mejor match en 'out'
        if best and best[1] >= threshold:                      # Si es lo suficientemente similar
            match_str, score, idx = best                       # Desempaqueta (texto, score, índice)
            if len(cand) > len(match_str):                     # Mantiene la más larga (más informativa)
                out[idx] = cand
        else:
            if cand.lower() not in (x.lower() for x in out):   # Evita duplicados exactos por minúscula
                out.append(cand)
    return out

def merge_and_pick_orgs(a: str, b: str, threshold: int) -> str:
    """
    Funde listas de afiliaciones de A y B (separadas por ';'),
    para cada fragmento extrae 'Organización, País', deduplica y concatena con '; '.
    """
    all_frags = []                                             # Lista combinada de fragmentos
    for source in (a or "", b or ""):                          # Recorre ambas fuentes
        all_frags.extend([x.strip() for x in str(source).split(';') if x.strip()])  # Divide por ';'

    picked = []                                                # Organizaciones priorizadas (con país)
    for frag in all_frags:
        org = pick_primary_org(frag)                           # Extrae "Org, País"
        if org:
            picked.append(org)

    deduped = dedupe_fuzzy(picked, threshold)                  # Deduplicación difusa
    return '; '.join(deduped)                                  # Une en una sola cadena

# ---------- 11) Verificación de columnas requeridas ----------
faltan = {COL_A, COL_B} - set(df.columns)                      # ¿Faltan columnas?
if faltan:
    raise ValueError(f"❌ Faltan columnas esperadas: {faltan}")# Lanza error explícito

# ---------- 12) Calcular columna de salida "Combined_universities" ----------
df["Combined_universities"] = df.apply(                        # Aplica por fila
    lambda r: merge_and_pick_orgs(r.get(COL_A, ""), r.get(COL_B, ""), DEDUPE_THRESHOLD),
    axis=1
)

# ---------- 13) Métrica rápida (control de calidad) ----------
total = len(df)                                                # Total de filas
vacias = int((df["Combined_universities"].str.strip() == "").sum())  # Filas sin resultado
print(f"Filas: {total:,} | Vacías en Combined_universities: {vacias:,} ({vacias/max(total,1):.2%})")  # KPI

# ---------- 14) Guardar CSV ----------
df.to_csv(OUT, index=False, encoding="utf-8")                  # Exporta sin índice, UTF-8
print(f"📄 Guardado: {OUT}")                                   # Confirma ruta de salida
