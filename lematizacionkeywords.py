import pandas as pd
import re, unicodedata
import spacy
from rapidfuzz import fuzz, process

# --- 0) Carga tu modelo spaCy ---
nlp = spacy.load('en_core_web_sm')

# --- 1) Tus funciones de limpieza y lematización (igual que antes) ---
def limpieza_basica(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto) \
                        .encode('ascii','ignore') \
                        .decode('utf-8','ignore')
    texto = texto.replace('(', '').replace(')', '')
    #texto = texto.replace('-', ' ')
    texto = re.sub(r"[^a-z0-9'\s\-/]", '', texto)
    return re.sub(r'\s+', ' ', texto).strip()
# Lista de sustantivos que NO se lematizan, se dejan tal cual
EXCEPTION_NOUNS = {
    "data",
    "children",
    "media",
    "criteria",
    "covid-19",
    "e-learning",
}

def normalizar_texto_spacy(texto: str, remove_stopwords: bool = False) -> str:
    texto = limpieza_basica(texto)
    doc = nlp(texto)
    out = []
    for token in doc:
        if remove_stopwords and token.is_stop:
            continue
        if token.is_punct or token.is_space:
            continue

        orig = token.text
        lema = token.lemma_

        if token.tag_ == "VBG":
            # gerundios se mantienen
            out.append(orig)
        elif token.pos_ == "NOUN" and orig.lower() in EXCEPTION_NOUNS:
            # excepciones de noun: no lematizar
            out.append(orig)
        else:
            # resto de casos: usar el lema (singulariza plurales)
            out.append(lema)

    return " ".join(out)

def normalizar_keywords_columna(celda: str) -> str:
    if not isinstance(celda, str): 
        return ""
    terms = [t.strip() for t in celda.split(';') if t.strip()]
    return ";".join(normalizar_texto_spacy(t) for t in terms)

# --- 2) Función de score combinado de RapidFuzz ---

# --- 3) Carga y preprocesa tu CSV ---

#ruta = "G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv"
#r"G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\datawos_scopus.csv"
 
ruta= r"G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\datawos_scopus.csv"
 


df = pd.read_csv(ruta).fillna("")


# --- 2) Definir los scorers disponibles ---
SCORERS = {
    'ratio':       fuzz.ratio,
    'partial':     fuzz.partial_ratio,
    'token_sort':  fuzz.token_sort_ratio,
    'token_set':   fuzz.token_set_ratio,
    'WRatio':   fuzz.WRatio
    
}

# --- 3) Función de agrupamiento con bandera “method” ---
def agrupar_rápido(
    cell: str,
    vocab: list,
    thresh: int = 85,
    method: str = 'token_set'
) -> str:
    """
    - cell: keywords separadas por ';'
    - vocab: lista global de términos canónicos (en minúsculas)
    - thresh: umbral 0–100
    - method: 'ratio' | 'partial' | 'token_sort' | 'token_set'
    """
    scorer = SCORERS.get(method, fuzz.token_set_ratio)
    seen, out = set(), []
    for term in [t.strip().lower() for t in cell.split(';') if t.strip()]:
        # excluyo el término idéntico para no emparejar con sí mismo
        choices = [v for v in vocab if v != term]
        # extraigo top-2 para poder saltarme el idéntico
        matches = process.extract(
            query=term,
            choices=choices,
            scorer=scorer,
            score_cutoff=thresh,

            limit=None   
        )
        
        # si hay segundo match y supera threshold, lo elijo
     
        if matches:
                           # 3) Escoge el más largo (max len de cadena)
            elegido = min(matches, key=lambda x: len(x[0]))[0]
        else:
            elegido = term   
     
        if elegido not in seen:
            seen.add(elegido)
            out.append(elegido)
    # dedup interno preservando orden
    return ';'.join(out)

def contar_unicos(col, df_input):
    # Drop NA, split por “;”, strip y filtrar vacíos
    all_keywords = (
        df_input[col]
        .dropna()
        .str.split(';')
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
    )
    return all_keywords.unique().shape[0]

# Conteo único antes de normalizar
print("Antes de normalizar:")
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")
# 4) Normaliza por separado Author y Index
df['Author Keywords'] = df['Author Keywords'].apply(normalizar_keywords_columna)
df['Index Keywords']  = df['Index Keywords'].apply(normalizar_keywords_columna)
print("\nDespués de normalizar y lematizacion:")
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")

# 5) Unifica ambas en All Keywords

# --- 5) Construye vocabulario global ---
def construir_vocab(df, col):
    return sorted({
        kw.strip().lower()
        for cell in df[col]
        for kw in cell.split(';') if kw.strip()
    })

vocab_global = sorted(
    set(construir_vocab(df,'Author Keywords') +
        construir_vocab(df,'Index Keywords'))
)

# --- 6) Aplica el agrupamiento con diferentes métodos ---
# Ratio puro


# Token sort ratio
df['Author Keywords'] = df['Author Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=98, method='WRatio'))
df['Index Keywords'] = df['Index Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=98, method='WRatio'))
print("\nDespués de WRatio:")
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")



# Conteo único antes de normalizar

# 7) Aplica fuzzy grouping a las tres columnas



# 9) Guarda el resultado
out = r"G:\\Mi unidad\\2025\\Master Espinoza Carrasco Alex Steven\\data\\datawos_scopuslematizafinal.csv"

df.to_csv(out, index=False)
print("Resultado guardado en:", out)
'''

# Token set ratio (default recomendado)
df['Author Keyword'] = df['Author Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=95, method='token_set'))
df['Index Keywords'] = df['Index Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=95, method='token_set'))
    
print("\nDespués de Token set ratio:")

for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")
df['AK_fuzzy_ratio'] = df['Author Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=90, method='ratio'))
df['IK_fuzzy_ratio'] = df['Index Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=90, method='ratio'))

# Partial ratio
df['Author Keywords'] = df['Author Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=95, method='partial'))
df['Author Keywords'] = df['Index Keywords'] \
    .apply(lambda c: agrupar_rápido(c, vocab_global, thresh=95, method='partial'))
print("\nDespués de ratio:")
for col in ['Index Keywords', 'Author Keywords']:
    print(f"  {col}: {contar_unicos(col, df)} keywords únicas")

'''