import pandas as pd
import re
from rapidfuzz import fuzz, process

# =====================================================
# 1. RUTAS
# =====================================================
ruta = r'G:\Mi unidad\2025\codigos bibliometria NPL'

archivo_maestro   = ruta + r'\databrazilcuartiles.xlsx'
archivo_brazil    = ruta + r'\databrazil_unificado.xlsx'
archivo_salida    = ruta + r'\DATASET_FINAL_BIBLIOMETRIA.xlsx'

# =====================================================
# 2. CARGA DE DATOS
# =====================================================
df_master = pd.read_excel(archivo_maestro)
df_br     = pd.read_excel(archivo_brazil)

# =====================================================
# 3. FUNCIONES DE NORMALIZACIÓN (MISMAS REGLAS)
# =====================================================
def normalizar_issn(x):
    if pd.isna(x):
        return ""
    return re.sub(r'[^0-9X]', '', str(x).upper())

def normalizar_titulo(x):
    if pd.isna(x):
        return ""
    x = x.lower()
    x = re.sub(r'\(.*?\)', '', x)      # elimina (online), (print), años
    x = re.sub(r'[^a-z0-9 ]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

# =====================================================
# 4. NORMALIZACIÓN
# =====================================================
df_master['ISSN_N']  = df_master['Issn'].apply(normalizar_issn)
df_master['TITLE_N'] = df_master['Title'].apply(normalizar_titulo)

df_br['ISSN_N']  = df_br['ISSN'].apply(normalizar_issn)
df_br['TITLE_N'] = df_br['titulo'].apply(normalizar_titulo)

# =====================================================
# 5. MATCHING BRASIL (ISSN → TÍTULO)
# =====================================================
br_titles = df_br['TITLE_N'].dropna().tolist()

def obtener_info_brasil(row, threshold=90):
    issn = row['ISSN_N']
    title = row['TITLE_N']

    # --- ISSN ---
    if issn:
        match = df_br[df_br['ISSN_N'] == issn]
        if not match.empty:
            fila = match.iloc[0]
            return (
                fila['Estrato'],
                fila['area'],
                'ISSN'
            )

    # --- TÍTULO ---
    if title:
        result = process.extractOne(
            title,
            br_titles,
            scorer=fuzz.WRatio
        )
        if result:
            _, score, idx = result
            if score >= threshold:
                fila = df_br.iloc[idx]
                return (
                    fila['Estrato'],
                    fila['area'],
                    'TITLE'
                )

    return (None, None, 'NONE')

# =====================================================
# 6. APLICAR MATCHING BRASIL
# =====================================================
df_master[
    ['BR_Estrato', 'BR_Area', 'BR_Match_Method']
] = df_master.apply(
    lambda r: pd.Series(obtener_info_brasil(r)),
    axis=1
)

# =====================================================
# 7. LIMPIEZA FINAL DE ÁREAS
# =====================================================
def limpiar_area(area):
    if pd.isna(area):
        return None
    area = str(area).strip()
    area = re.sub(r",\s*$", "", area)
    area = re.sub(r"\s*;\s*", "; ", area)
    return area

df_master['BR_Area'] = df_master['BR_Area'].apply(limpiar_area)

# =====================================================
# 8. GUARDAR DATASET FINAL
# =====================================================
df_master.to_excel(archivo_salida, index=False)

print("INTEGRACIÓN COMPLETADA CORRECTAMENTE")
print(f"Registros finales: {len(df_master)}")

print("\nDistribución matching Brasil:")
print(df_master['BR_Match_Method'].value_counts())

print(f"\nArchivo final guardado en:\n{archivo_salida}")
