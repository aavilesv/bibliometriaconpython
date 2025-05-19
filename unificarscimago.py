import pandas as pd
import os
import re

# Ruta de la carpeta que contiene los archivos
folder_path = r"G:\\Mi unidad\\Maestría en inteligencia artificial\\Master Angelo Aviles\\bibliometria 2 scopus\\data"

# Lista de archivos
files = [
    "datascimago2013.csv",
    "datascimago2014.csv",
    "datascimago2015.csv",
    "datascimago2016.csv",
    "datascimago2017.csv",
    "datascimago2018.csv",
    "datascimago2019.csv",
    "datascimago2020.csv",
    "datascimago2021.csv",
    "datascimago2022.csv",
    "datascimago2023.csv",
    "datascimago2024.csv"
]

# Columnas que queremos conservar (asegúrate de incluir 'Year')
columns_to_keep = [
    "Issn", "SJR Best Quartile", "H index", "Country", "Region", "SJR", 
    "Publisher", "Coverage", "Categories", "Areas", "Year"
]

dataframes = []

for file in files:
    # Cargar el CSV
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, sep=";")

    # Extraer el año del nombre de archivo (cuatro dígitos)
    match = re.search(r'(\d{4})', file)
    if match:
        year = int(match.group(1))
    else:
        raise ValueError(f"No se pudo extraer el año de '{file}'")

    # Añadir la columna Year
    df['Year'] = year

    # Seleccionar sólo las columnas que queremos
    df = df[columns_to_keep]

    dataframes.append(df)

# Concatenar todo en un solo DataFrame
merged_data = pd.concat(dataframes, ignore_index=True)

# Guardar resultado
output_path = os.path.join(folder_path, "scimago_unificado.csv")
merged_data.to_csv(output_path, sep=";", index=False)

print("Archivo unificado generado en:", output_path)
