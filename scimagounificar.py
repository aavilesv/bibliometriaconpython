import pandas as pd
import glob
import os
import re

path = r"G:\Mi unidad\scimago"
files = glob.glob(os.path.join(path, "scimagojr*.csv"))

dfs = []

for file in files:
    year = int(os.path.basename(file).split()[-1].replace(".csv", ""))

    df = pd.read_csv(
        file,
        sep=";",
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip"
    )

    # 1) Eliminar espacios raros en nombres
    df.columns = [c.strip() for c in df.columns]

    # 2) Renombrar Total Docs. (YYYY) -> Total Docs
    # (solo el del año correspondiente)
    col_year = f"Total Docs. ({year})"
    if col_year in df.columns:
        df = df.rename(columns={col_year: "Total Docs."})

    # 3) Si existe un Publisher duplicado (Publisher.1), lo eliminamos/normalizamos
    # Regla: si "Publisher" está vacío y "Publisher.1" tiene valor, usar Publisher.1
    if "Publisher.1" in df.columns:
        if "Publisher" in df.columns:
            df["Publisher"] = df["Publisher"].fillna("").astype(str).str.strip()
            df["Publisher.1"] = df["Publisher.1"].fillna("").astype(str).str.strip()
            df.loc[df["Publisher"].eq(""), "Publisher"] = df.loc[df["Publisher"].eq(""), "Publisher.1"]
        df = df.drop(columns=["Publisher.1"])

    # 4) Añadir year
    df["year"] = year

    dfs.append(df)

scimago_longitudinal = pd.concat(dfs, ignore_index=True)

# Guardar
out = os.path.join(path, "scimago_2013_2024_longitudinal.csv")
scimago_longitudinal.to_csv(out, index=False)

print("OK ->", out)
print("Columnas:", len(scimago_longitudinal.columns))
print(scimago_longitudinal["year"].value_counts().sort_index())
