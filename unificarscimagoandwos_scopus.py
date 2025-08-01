import pandas as pd
from rapidfuzz import process, fuzz
import re
import seaborn as sns
import matplotlib.pyplot as plt
##  usar por favor esto para poder validar todo from rapidfuzz import fuzz, process
# Rutas de los archivos SCImago y Scopus
scimago_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\scimago_unificado.csv"
wos_scopus_path =  "G:\\Mi unidad\\Master en administración y empresas\\articulo 3\\data\\datawos_scopus.csv"



# Cargar ambos archivos CSV con separador ";"
scimago_data = pd.read_csv(scimago_path, sep=";")
wos_scopus = pd.read_csv(wos_scopus_path)
scimago_data = scimago_data.rename(columns={
               'Title' : 'Source title'
               })


scimago_data['Source title'] =scimago_data['Source title'].str.replace(r'\([^)]*\)', '', regex=True).str.strip()
def clean_categories(cat_str):
    # 1) Elimina todo lo que esté entre paréntesis, incluidas múltiples ocurrencias
    no_par = re.sub(r'\([^)]*\)', '', cat_str)
    # 2) Divide por “;”, recorta espacios y descarta entradas vacías
    parts = [p.strip() for p in no_par.split(';') if p.strip()]
    # 3) Une con un único “; ” entre categorías
    return '; '.join(parts)

# Aplicarlo a la columna
scimago_data['Categories'] = scimago_data['Categories'].apply(clean_categories)
# Normalizar ISSN en SCImago para manejar múltiples valores
scimago_expanded = scimago_data.assign(Issn=scimago_data['Issn'].str.split(',')).explode('Issn')

scimago_expanded['Issn'] = scimago_expanded['Issn'].str.strip()  # Eliminar espacios adicionales en ISSN
total = len(wos_scopus)

# 1) Merge por ISSN
by_issn = pd.merge(
    wos_scopus,
    scimago_expanded,
    how="left",
    left_on=["ISSN", "Year"],
    right_on=["Issn", "Year"],
    suffixes=("", "_scimago")
)
# 2) Aquellos sin match por ISSN
sin_issn = by_issn[by_issn['Issn'].isna()].drop(
    columns=scimago_expanded.columns.difference(['Source title', 'Year']),
    errors='ignore'
)
# 1) Prepara la lista de títulos únicos en SCImago

scimago_titles = scimago_expanded['Source title'].unique().tolist()



def best_match(title, threshold=90):
    title = title.strip().lower()
    result = process.extractOne(query=title, choices=scimago_titles, scorer=fuzz.WRatio)
    if result:
        match, score, _idx = result
        if score >= threshold:
            return match
    return None

# 3) Aplica el matching difuso a cada registro sin ISSN
sin_issn['Source title'] = sin_issn['Source title'].apply(lambda t: best_match(t, threshold=90))
# 4) Filtra sólo los que encontraron un match aceptable
to_merge = sin_issn.dropna(subset=['Source title'])


# 3) Merge por título solo para los sin ISSN
by_title = pd.merge(
    to_merge,
    scimago_expanded,
    how="left",
    
    left_on=["Source title", "Year"],
    right_on=["Source title", "Year"],
    suffixes=("", "_scimago")
)

# 4) Separar los dos grupos de matches
matched_issn  = by_issn[~by_issn['Issn'].isna()]
matched_title = by_title[~by_title['Issn'].isna()]

# 5) Concatenar solo los que sí matchearon
matched = pd.concat([matched_issn, matched_title], ignore_index=True)

# 6) Métricas
matched_count   = len(matched)
unmatched_count = total - matched_count
pct_match       = matched_count / total * 100

print(f"Total Web of Science y Scopus:             {total}")
print(f"Total matched Scimago:    {matched_count}")
print(f"Total unmatched Scimago:  {unmatched_count}")
print(f"% coincidencias:          {pct_match:.2f}%")

#output_path =  "G:\\Mi unidad\\Master en administración y empresas\\articulo 3\\data\\datawos_scopus_scimago.csv"

#print("Archivo combinado guardado en:", output_path)
#matched.to_csv(output_path, sep=";", index=False)

# 2) Construir el DataFrame “total”:
unmatched = wos_scopus.loc[~wos_scopus.index.isin(matched.index)]
all_with_info = pd.concat([matched, unmatched], ignore_index=True)

print(f"Guardado total: {len(all_with_info)} registros")
print(f"Guardado matched: {len(matched)} registros")
all_with_info['Cited by'] = pd.to_numeric(all_with_info['Cited by'],
                                          errors='coerce')           # NaN si falta
all_with_info['SJR Best Quartile'] = (
        all_with_info['SJR Best Quartile']
        .str.upper().str.strip()                                     # “q1 ” → “Q1”
)

# Ordenar la categoría
all_with_info['SJR Best Quartile'] = pd.Categorical(
        all_with_info['SJR Best Quartile'],
        categories=['Q1','Q2','Q3','Q4'],
        ordered=True
)

# ------------------------------------------------------------------
# 1) Tabla resumen por cuartil
# ------------------------------------------------------------------
summary = (all_with_info
           .dropna(subset=['SJR Best Quartile'])      # evita cuartil vacío
           .groupby('SJR Best Quartile')
           .agg(Articles   = ('Title',    'count'),
                MedianCites=('Cited by', 'median'),
                MeanCites  = ('Cited by', 'mean'),
                IQR_low    = ('Cited by', lambda x: x.quantile(.25)),
                IQR_high   = ('Cited by', lambda x: x.quantile(.75)))
           .reset_index())

print(summary.to_string(index=False))

# ------------------------------------------------------------------
# 2) Porcentaje de Q1
# ------------------------------------------------------------------
q1_share = (summary.loc[summary['SJR Best Quartile']=='Q1','Articles']
                    .iat[0] / len(all_with_info) * 100)
print(f"\nQ1 share: {q1_share:.1f}%")

# ------------------------------------------------------------------
# 3) Box-plot (con puntos superpuestos)
# ------------------------------------------------------------------
plt.figure(figsize=(7,5))
sns.boxplot(data=all_with_info, x='SJR Best Quartile', y='Cited by',
            order=['Q1','Q2','Q3','Q4'], palette='Set2')
sns.stripplot(data=all_with_info, x='SJR Best Quartile', y='Cited by',
              order=['Q1','Q2','Q3','Q4'], color='black', size=3, alpha=.3)
plt.yscale('log')
plt.title('Citation distribution by SJR quartile', weight='bold')
plt.xlabel('SJR Best Quartile')
plt.ylabel('Citations (log scale)')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4) (Opcional) Exportar tabla a Excel
# ------------------------------------------------------------------
summary.to_excel('SJR_quartile_stats.xlsx', index=False)

all_with_info['Index Keywords'] = all_with_info['Areas'].str.replace('', '')
all_with_info['Author Keywords'] = all_with_info['Categories'].str.replace('', '')

columns_to_drop = [
    'Source title_scimago',
    'Issn', 
    'H index',
    'SJR Best Quartile',
    'Country',
    'SJR Best Quartile',
    'Region',
    'SJR',
    'Publisher',
        'Coverage',
    'Categories',
    'Areas',
]
#all_with_info = all_with_info.drop(columns=columns_to_drop)
# 3) Guardar el total
all_with_info.to_csv("G:\\Mi unidad\\Master en administración y empresas\\articulo 3\\data\\datawos_scopus_scimago.csv", sep=";", index=False)


