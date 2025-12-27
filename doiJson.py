#!/usr/bin/env python3
"""
Descarga metadata de CrossRef para una lista de DOIs
y la guarda en archivos JSON (común y JSONL).
"""

import json
import requests
from urllib.parse import quote

# === TUS DOIS ===
DOIS = [
    "10.59282/reincisol.V4(8)5780-5813",
    "10.1080/00393541.2014.11518931",
    "10.1111/jade.12173",
    "10.1002/berj.3546",
    "10.1080/00131725.2014.971991",
    "10.47553/rifop.v99i38.1.104030",
    "10.1080/00220272.2020.1720299",
    "10.1386/eta_00085_2",
    "10.56300/uanj1022",
    "10.1016/j.tsc.2021.100861",
    "10.3102/0034654314540477",
    "10.1080/15290824.2014.922188",
    "10.1007/s43545-022-00513-6",
    "10.1007/s13384-012-0051-2",
    "10.14221/ajte.2016v41n5.2",
    "10.1177/1477971419846640",
    "10.1080/10632921.2013.775980",
    "10.1386/eta.13.3.395_1",
    "10.1002/berj.4029",
    "10.1007/s11192-015-1765-5",
    "10.7203/eari.13.22984",
    "10.1080/15505170.2016.1219890",
    "10.26209/ijea25n19",
    "10.1007/s11192-019-03213-w",
    "10.5209/aris.72439",
    "10.1386/eta.13.3.333_1",
    "10.1080/09500693.2017.1333656",
    "10.1007/s11192-009-0146-3",
]

def get_crossref_metadata(doi: str):
    """Obtiene metadata desde CrossRef para un DOI dado y devuelve el 'message'."""
    url = f"https://api.crossref.org/works/{quote(doi)}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] No se pudo obtener metadata de CrossRef para {doi}: {e}")
        return None

    data = resp.json()
    return data.get("message", {})

def main():
    all_records = []  # para el JSON "normal"

    # Abrimos el JSONL en modo escritura
    jsonl_filename = "crossref_metadata.jsonl"
    jsonl_file = open(jsonl_filename, "w", encoding="utf-8")

    for doi in DOIS:
        doi = doi.strip()
        if not doi:
            continue

        print(f"\n=== Procesando DOI: {doi} ===")
        meta = get_crossref_metadata(doi)
        if not meta:
            print(f"[ERROR] No se obtuvo metadata para {doi}.")
            continue

        # Estructura estándar que guardaremos
        record = {
            "doi": meta.get("DOI", doi),
            "metadata": meta
        }

        # Acumular para el JSON "grande"
        all_records.append(record)

        # Escribir una línea en JSONL
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    jsonl_file.close()
    print(f"\n[OK] Guardado archivo JSONL: {jsonl_filename}")

    # Guardar todo junto en un JSON con indentado
    json_filename = "crossref_metadata.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"[OK] Guardado archivo JSON: {json_filename}")
    print(f"Total de registros guardados: {len(all_records)}")

if __name__ == "__main__":
    main()
