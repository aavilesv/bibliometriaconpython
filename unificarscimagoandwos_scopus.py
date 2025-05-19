import pandas as pd

# Rutas de los archivos SCImago y Scopus
scimago_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\scimago_unificado.csv"
scopus_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\datafinal.csv"

# Cargar ambos archivos CSV con separador ";"
scimago_data = pd.read_csv(scimago_path, sep=";")
scopus_data = pd.read_csv(scopus_path)

# Normalizar ISSN en SCImago para manejar múltiples valores
scimago_expanded = scimago_data.assign(Issn=scimago_data['Issn'].str.split(',')).explode('Issn')
scimago_expanded['Issn'] = scimago_expanded['Issn'].str.strip()  # Eliminar espacios adicionales en ISSN
# Guardar el total de registros originales de Scopus
total_scopus_records = len(scopus_data)
# Realizar la unión basada en ISSN y Year, copiando todas las columnas de SCImago en Scopus
merged_data = pd.merge(scopus_data, scimago_expanded, how="left", left_on=["ISSN", "Year"], right_on=["Issn", "Year"])
# Contar cuántos registros no encontraron coincidencia en SCImago
unmatched_records = merged_data['Issn'].isna().sum()

# Contar cuántos registros se unieron exitosamente (es decir, con coincidencia en SCImago)
matched_records = total_scopus_records - unmatched_records

# Calcular el porcentaje de coincidencias
match_percentage = (matched_records / total_scopus_records) * 100

# Guardar el archivo resultante con la información combinada
output_path = "G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data\\dataunificada.csv"
print("Archivo combinado guardado en:", output_path)
merged_data.to_csv(output_path, sep=";", index=False)
# Filtrar solo los registros que encontraron coincidencia en SCImago
matched_data = merged_data[~merged_data['Issn'].isna()]

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