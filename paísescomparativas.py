INPUT  = r"G:\\Mi unidad\\2025\\Master Estefania Landires\\datawos_scopus.csv"
OUTPUT = r"G:\\Mi unidad\\2025\\Master Estefania Landires\\datawos_scopusinstitucion.csv"
# ──────────────────────────────────────────────────────────
# LIBRERÍAS
# pip install pandas rapidfuzz
# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────
# NORMALIZACIÓN DE AFILIACIONES (RapidFuzz + eliminación de [ … ])
# Requisitos: pandas, rapidfuzz  →  pip install pandas rapidfuzz
# ──────────────────────────────────────────────────────────
import re, unicodedata, pandas as pd
from rapidfuzz import fuzz

# ──────────────── CONFIGURA TUS RUTAS ─────────────────────
INPUT  = r"G:\\Mi unidad\\2025\\Master Estefania Landires\\datawos_scopus.csv"
OUTPUT = r"G:\\Mi unidad\\2025\\Master Estefania Landires\\datawos_scopusinstitucion.csv"

# Keywords para detectar instituciones (minúsculas, sin tildes)
KEYWORDS = ['univ', 'university', 'fac', 'faculty', 'inst', 'institute',
            'dept', 'department', 'college', 'school', 'ctr', 'center']

THRESH = 20   # Baja para fusionar más, sube para fusionar menos

# ─────────────────── FUNCIONES AUXILIARES ─────────────────
def strip_acc(text: str) -> str:
    """Quita diacríticos."""
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c))

# Borra cualquier cosa entre '[' y ']', incluso con saltos de línea
remove_brackets = lambda t: re.sub(r'\[.*?]', '', t, flags=re.DOTALL)

def clean_fragment(frag: str) -> str:
    """Limpieza mínima, lower y sin acentos."""
    frag = strip_acc(frag).lower()
    return re.sub(r'\s+', ' ', frag).strip()

def has_kw(frag: str) -> bool:
    return any(k in frag for k in KEYWORDS)

def same_institution(a: str, b: str) -> bool:
    """
    Dos fragmentos son la misma institución si:
      1) partial_ratio == 100   → a es subcadena exacta de b (o vice-versa)
      2) O partial_ratio ≥ THRESH  y  comparten alguna KEYWORD
    """
    score = fuzz.token_set_ratio(a, b)
    return score == 100 or (score >= THRESH and (has_kw(a) or has_kw(b)))

# ──────────── NORMALIZACIÓN FILA A FILA ────────────
def normalize_row(text_union: str) -> str:
    """
    ① Quita bloques [ … ]
    ② Divide por ';', limpia y fusiona variantes (conserva orden).
    """
    text_union = remove_brackets(text_union)

    raw_frags = [f.strip() for f in text_union.split(';') if f.strip()]
    cln_frags = [clean_fragment(f) for f in raw_frags]

    reps, mapped = [], []

    for frag in cln_frags:
        idx = next((i for i, rep in enumerate(reps) if same_institution(frag, rep)),
                   None)

        if idx is None:                     # nuevo grupo
            reps.append(frag)
            mapped.append(frag)
        else:                               # ya existe
            if len(frag) > len(reps[idx]):  # conserva la más larga
                reps[idx] = frag
            mapped.append(reps[idx])

    return '; '.join(mapped)

# ───────────────────────── MAIN ──────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(INPUT, encoding='utf-8-sig')

    # Concatenar columnas sin alterar posiciones
    df['union'] = (df['Affiliations'].fillna('') + ' ' +
                   df['Authors with affiliations'].fillna(''))

    # Aplicar normalización
    df['Affiliations_Normalized'] = df['union'].apply(normalize_row)

    # Guardar
    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print("✅ Normalización terminada. Archivo guardado en:\n", OUTPUT)
