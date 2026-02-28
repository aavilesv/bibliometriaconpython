import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, process

# =====================================================
# 1. RUTAS
# =====================================================
ruta = r'G:\\Mi unidad\\2025\\codigos bibliometria NPL\\'

archivo_data1  = ruta + r'\\datacomunication.xlsx'                 # Scimago
archivo_data2  = ruta + r'\\ext_list_Dec_2025.xlsx'          # Scopus Source List
archivo_data3  = ruta + r'\\UNIFICADO_WOS_JCR_2024_FINAL.xlsx'  # WoS/JCR

archivo_salida = ruta + r'\\data12_actualizado_filtrado2.xlsx'

# =====================================================
# 2. CARGA DE DATOS
# =====================================================
df1 = pd.read_excel(archivo_data1)
df2 = pd.read_excel(archivo_data2)
df3 = pd.read_excel(archivo_data3)

# =====================================================
# 3. FUNCIONES DE NORMALIZACIÓN
# =====================================================
def normalizar_issn(x):
    if pd.isna(x):
        return ""
    return re.sub(r'[^0-9X]', '', str(x).upper())

def normalizar_titulo(x):
    if pd.isna(x):
        return ""
    x = x.lower()
    x = re.sub(r'[^a-z0-9 ]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

# =====================================================
# 4. NORMALIZACIÓN SCOPUS
# =====================================================
df2['ISSN_N']  = df2['ISSN'].apply(normalizar_issn)
df2['EISSN_N'] = df2['EISSN'].apply(normalizar_issn)
df2['TITLE_N'] = df2['Source Title'].apply(normalizar_titulo)

scopus_titles = df2['TITLE_N'].dropna().tolist()

# =====================================================
# 5. FUNCIÓN MATCHING SCOPUS (ISSN → TÍTULO)
# =====================================================
def obtener_estado_scopus(row, threshold=90):
    issn_val  = row.get('Issn', '')
    title_val = row.get('Title', '')

    # --- ISSN ---
    issns = [
        normalizar_issn(c)
        for c in str(issn_val).replace(';', ',').split(',')
        if normalizar_issn(c)
    ]

    for issn in issns:
        match = df2[
            (df2['ISSN_N'] == issn) |
            (df2['EISSN_N'] == issn)
        ]
        if not match.empty:
            return match.iloc[0]['Active or Inactive'], 'ISSN'

    # --- TÍTULO ---
    title_n = normalizar_titulo(title_val)
    if title_n:
        result = process.extractOne(
            title_n,
            scopus_titles,
            scorer=fuzz.WRatio
        )
        if result:
            _, score, idx = result
            if score >= threshold:
                return df2.iloc[idx]['Active or Inactive'], 'TITLE'

    return 'Not in Scopus', 'NONE'

# =====================================================
# 6. APLICAR MATCHING SCOPUS
# =====================================================
df1[['Scopus_Status', 'Scopus_Match_Method']] = df1.apply(
    lambda r: pd.Series(obtener_estado_scopus(r)),
    axis=1
)

# =====================================================
# 7. LISTA NEGRA DE CATEGORÍAS (SCIMAGO)
# =====================================================
categorias_excluir = [
    # deja vacío o agrega categorías si lo necesitas
]

categorias_excluir = [c.lower() for c in categorias_excluir]

def contiene_categoria_excluida(categorias):
    if pd.isna(categorias):
        return False
    txt = categorias.lower()
    for cat in categorias_excluir:
        if re.search(rf"\b{re.escape(cat)}\b", txt):
            return True
    return False

# =====================================================
# 8. FILTRADO SCOPUS + CATEGORÍAS
# =====================================================
df_filtrado = df1[~df1['Categories'].apply(contiene_categoria_excluida)]
df_filtrado = df_filtrado.drop_duplicates()

# =====================================================
# 9. NORMALIZACIÓN WoS / JCR
# =====================================================
df3['ISSN_N']  = df3['ISSN'].apply(normalizar_issn)
df3['EISSN_N'] = df3['eISSN'].apply(normalizar_issn)
df3['TITLE_N'] = df3['Journal name'].apply(normalizar_titulo)

wos_titles = df3['TITLE_N'].dropna().tolist()

# =====================================================
# 10. FUNCIÓN MATCHING WoS (ISSN → TÍTULO)
# =====================================================
def obtener_estado_wos(row, threshold=95):
    issn_val  = row.get('Issn', '')
    title_val = row.get('Title', '')

    # --- ISSN ---
    issns = [
        normalizar_issn(c)
        for c in str(issn_val).replace(';', ',').split(',')
        if normalizar_issn(c)
    ]

    for issn in issns:
        match = df3[
            (df3['ISSN_N'] == issn) |
            (df3['EISSN_N'] == issn)
        ]
        if not match.empty:
            fila = match.iloc[0]
            return (
                'In WoS',
                'ISSN',
                fila['Edition'],
                fila['2024 JIF'],
                fila['JIF Quartile']
            )

    # --- TÍTULO ---
    title_n = normalizar_titulo(title_val)
    if title_n:
        result = process.extractOne(
            title_n,
            wos_titles,
            scorer=fuzz.WRatio
        )
        if result:
            _, score, idx = result
            if score >= threshold:
                fila = df3.iloc[idx]
                return (
                    'In WoS',
                    'TITLE',
                    fila['Edition'],
                    fila['2024 JIF'],
                    fila['JIF Quartile']
                )

    return ('Not in WoS', 'NONE', None, None, None)

# =====================================================
# 11. APLICAR MATCHING WoS
# =====================================================
df_filtrado[
    ['WoS_Status', 'WoS_Match_Method', 'WoS_Edition', 'WoS_JIF_2024', 'WoS_Q']
] = df_filtrado.apply(
    lambda r: pd.Series(obtener_estado_wos(r)),
    axis=1
)

# =====================================================
# 12. GUARDAR RESULTADO FINAL
# =====================================================
df_filtrado.to_excel(archivo_salida, index=False)

print("Proceso completado correctamente.")
print(f"Registros iniciales: {len(df1)}")
print(f"Registros finales: {len(df_filtrado)}")

print("\nDistribución Scopus:")
print(df_filtrado['Scopus_Match_Method'].value_counts())

print("\nDistribución WoS:")
print(df_filtrado['WoS_Match_Method'].value_counts())

print(f"\nArchivo guardado en: {archivo_salida}")
