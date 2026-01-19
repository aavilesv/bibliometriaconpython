import os
import glob
import pandas as pd
import re

# ================= CONFIGURACIÓN =================
CARPETA = r"G:\Mi unidad\2025\revistas de woss"
SALIDA = os.path.join(CARPETA, "UNIFICADO_WOS_JCR_2024_FINAL.xlsx")
# =================================================

COLUMNAS_OBJETIVO = [
    "Journal name",
    "JCR Abbreviation",
    "Publisher",
    "ISSN",
    "eISSN",
    "Category",
    "Edition",
    "Total Citations",
    "2024 JIF",
    "JIF Quartile",
    "2024 JCI",
    "% of Citable OA"
]

ISSN_REGEX = re.compile(r"^\d{4}-\d{3}[\dX]$")


def detectar_fila_header(df_raw):
    """
    Busca la fila que contiene 'ISSN' y la usa como encabezado real
    """
    for i in range(len(df_raw)):
        fila = df_raw.iloc[i].astype(str).str.strip().str.lower()
        if "issn" in fila.values:
            return i
    return None


def issn_valido(valor):
    if pd.isna(valor):
        return False
    return bool(ISSN_REGEX.match(str(valor).strip()))


def limpiar_category(valor):
    if pd.isna(valor):
        return []
    texto = str(valor).replace(",", ";")
    partes = [p.strip() for p in texto.split(";") if p.strip()]
    return list(dict.fromkeys(partes))


def unir_categorias(series):
    seen = set()
    out = []
    for lst in series:
        for c in lst:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return ";".join(out)


def main():
    archivos = glob.glob(os.path.join(CARPETA, "*.xlsx"))
    if not archivos:
        raise FileNotFoundError("No se encontraron archivos .xlsx")

    dfs = []

    for archivo in archivos:
        # Leer sin header
        df_raw = pd.read_excel(archivo, header=None, dtype=str)

        fila_header = detectar_fila_header(df_raw)
        if fila_header is None:
            print(f"⚠ No se detectó header en {os.path.basename(archivo)} — archivo omitido")
            continue

        # Reconstruir dataframe con header correcto
        df = pd.read_excel(
            archivo,
            header=fila_header,
            dtype=str
        )

        # Normalizar encabezados
        df.columns = [c.strip() for c in df.columns]

        # Quedarse solo con columnas objetivo
        df = df[[c for c in df.columns if c in COLUMNAS_OBJETIVO]]

        # Filtro estructural definitivo: ISSN o eISSN válidos
        df = df[
            df.get("ISSN", "").apply(issn_valido) |
            df.get("eISSN", "").apply(issn_valido)
        ]

        dfs.append(df)

    if not dfs:
        raise RuntimeError("Ningún archivo válido fue procesado")

    all_df = pd.concat(dfs, ignore_index=True)

    # Procesar categorías
    all_df["_cat_list"] = all_df["Category"].apply(limpiar_category)

    key_cols = [c for c in COLUMNAS_OBJETIVO if c != "Category"]
    all_df[key_cols] = all_df[key_cols].fillna("")

    resultado = (
        all_df
        .groupby(key_cols, as_index=False)
        .agg({"_cat_list": unir_categorias})
    )

    resultado["Category"] = resultado["_cat_list"]
    resultado = resultado.drop(columns="_cat_list")

    resultado = resultado[COLUMNAS_OBJETIVO]

    resultado.to_excel(SALIDA, index=False)

    print("PROCESO COMPLETADO CORRECTAMENTE")
    print(f"Archivo generado: {SALIDA}")
    print(f"Revistas únicas finales: {len(resultado):,}")


if __name__ == "__main__":
    main()
