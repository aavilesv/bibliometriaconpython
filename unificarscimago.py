import pandas as pd
from rapidfuzz import process, fuzz

import re
##  usar por favor esto para poder validar todo from rapidfuzz import fuzz, process
# Rutas de los archivos SCImago y Scopus
scimago_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\scimago_unificado.csv"
scopus_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\datafinal.csv"

# Cargar ambos archivos CSV con separador ";"
scimago_data = pd.read_csv(scimago_path, sep=";")
datasetw_s = pd.read_csv(scopus_path)

# Normalizar ISSN en SCImago para manejar múltiples valores
scimago_expanded = scimago_data.assign(Issn=scimago_data['Issn'].str.split(',')).explode('Issn')
scimago_expanded['Issn'] = scimago_expanded['Issn'].str.strip()  # Eliminar espacios adicionales en ISSN
# Guardar el total de registros originales de Scopus
total_scopus_records = len(datasetw_s)
# Realizar la unión basada en ISSN y Year, copiando todas las columnas de SCImago en Scopus



def merge_with_scimago(main_df: pd.DataFrame,
                       scimago_expanded: pd.DataFrame,
                       threshold: int = 90) -> pd.DataFrame:


       
 # 3) Agrupar para un registro único por (Issn_clean, Year)
    scimago_unique = (
        scimago_expanded
        .dropna(subset=['Issn', 'Year'])
        .sort_values(['Issn', 'Year'])
        .drop_duplicates(subset=['Issn', 'Year'], keep='first')
        .copy()
    )
    # 3) Merge por ISSN limpio y Year
    merged = main_df.merge(
        scimago_unique ,
        how='left',
        left_on=['ISSN', 'Year'],
        right_on=['Issn', 'Year'],
        suffixes=('', '_sci')
    )

    # d) Limpiar paréntesis en Title_sci
    merged['Title'] = (
        merged.get('Title', pd.Series(dtype=str))
              .str.replace(r'\([^)]*\)', '', regex=True)
              .str.strip()
    )
    # e) Fuzzy match donde no hubo match exacto
    catalog = merged['Title'].dropna().unique().tolist()
    def fuzzy_fill(row):
        if pd.notna(row['Title']):
            return None
        best, score, _ = process.extractOne(
            row['Source title'],
            catalog,
            scorer=fuzz.token_sort_ratio
        )
        return best if score >= threshold else None
    merged['Title'] = merged.apply(fuzzy_fill, axis=1)


    # f) Construir Title_final
    merged['Source title'] = (
        merged['Source title']
        .str.replace(r'\([^)]*\)', '', regex=True)
        .str.strip()
    )
    merged['Title'] = (
        merged['Title']
              .fillna(merged['Title'])
              .fillna(merged['Source title'])
    )
  # g) Registrar diferencias
    merged['Title'] = merged['Title'].where(
        merged['Title'] != merged['Source title']
    )
   # h) Sobrescribir columnas en main_df
    main_df['Source title'] = merged['Title'].values
    

    return main_df
merged_data = merge_with_scimago(datasetw_s, scimago_expanded)



#merged_data = pd.merge(datasetw_s, scimago_expanded, how="left", left_on=["ISSN", "Year"], right_on=["Issn", "Year"])
# Contar cuántos registros no encontraron coincidencia en SCImago
unmatched_records = merged_data['ISSN'].isna().sum()

# Contar cuántos registros se unieron exitosamente (es decir, con coincidencia en SCImago)
matched_records = total_scopus_records - unmatched_records

# Calcular el porcentaje de coincidencias
match_percentage = (matched_records / total_scopus_records) * 100

# Guardar el archivo resultante con la información combinada
output_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\dataunificada.csv"
print("Archivo combinado guardado en:", output_path)
merged_data.to_csv(output_path, sep=";", index=False)
# Filtrar solo los registros que encontraron coincidencia en SCImago
matched_data = merged_data[~merged_data['ISSN'].isna()]

# Guardar el archivo resultante con solo los registros coincidentes
output_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\data_unificada_matched.csv"
matched_data.to_csv(output_path, sep=";", index=False)
print("Archivo combinado guardado en:", output_path)
# Resumen descriptivo
print("Resumen del proceso de unión de datos:")
print(f"Total de registros en el archivo Scopus: {total_scopus_records}")
print(f"Total de registros que encontraron coincidencia en SCImago: {matched_records}")
print(f"Total de registros que no encontraron coincidencia en SCImago: {unmatched_records}")
print(f"Porcentaje de coincidencias: {match_percentage:.2f}%")
print(f"Archivo combinado guardado en: {output_path}")