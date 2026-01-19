# -*- coding: utf-8 -*-
"""
crossref_fetch_one.py
Consulta la API pública de Crossref para un DOI y guarda resultados en JSON y CSV.

Uso:
    python crossref_fetch_one.py
    # opcional: python crossref_fetch_one.py 10.1097/md.0000000000038955
"""

import sys
import csv
import json
import time
import requests
from pathlib import Path

CROSSREF_WORKS_BASE = "https://api.crossref.org/works/"

def fetch_crossref_by_doi(doi: str, mailto: str | None = None, max_retries: int = 3, timeout: int = 20) -> dict:
    """Devuelve el JSON de Crossref para un DOI."""
    url = CROSSREF_WORKS_BASE + requests.utils.quote(doi)
    params = {}
    if mailto:
        params["mailto"] = mailto

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            # Manejo básico de rate limit (HTTP 429) o errores temporales
            if r.status_code == 429 and attempt < max_retries:
                wait = int(r.headers.get("Retry-After", "2"))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == max_retries:
                raise RuntimeError(f"Error al consultar Crossref ({url}): {e}") from e
            time.sleep(1.5 * attempt)
    raise RuntimeError("No se pudo obtener respuesta de Crossref.")

def first_or_none(seq, default=None):
    if isinstance(seq, list) and seq:
        return seq[0]
    return default

def join_names(authors: list) -> str:
    """Devuelve 'Apellido, Nombre; Apellido, Nombre; ...'."""
    if not isinstance(authors, list):
        return ""
    parts = []
    for a in authors:
        given = a.get("given", "")
        family = a.get("family", "")
        full = (family + (", " + given if given else "")).strip(", ").strip()
        if not full and a.get("name"):
            full = a["name"]
        parts.append(full)
    return "; ".join([p for p in parts if p])

def flatten_affiliations(authors: list) -> str:
    """Aglutina afiliaciones por autor en 'Autor: Afiliación1 | Afiliación2; ...'."""
    if not isinstance(authors, list):
        return ""
    rows = []
    for a in authors:
        name = (a.get("family", "") + (" " + a.get("given", "") if a.get("given") else "")).strip()
        if not name and a.get("name"):
            name = a["name"]
        affs = a.get("affiliation", [])
        aff_texts = [aff.get("name", "").strip() for aff in affs if aff.get("name")]
        if name or aff_texts:
            rows.append(f"{name}: " + " | ".join([t for t in aff_texts if t]))
    return "; ".join(rows)

def extract_year(msg: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        part = msg.get(key, {})
        date_parts = part.get("date-parts") or []
        if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            return date_parts[0][0]
    return None

def normalize_record(data: dict) -> dict:
    """Extrae campos comunes de la respuesta de Crossref."""
    msg = data.get("message", {})
    record = {
        "doi": msg.get("DOI"),
        "title": first_or_none(msg.get("title")),
        "subtitle": first_or_none(msg.get("subtitle")),
        "journal": first_or_none(msg.get("container-title")),
        "publisher": msg.get("publisher"),
        "type": msg.get("type"),
        "issn": "; ".join(msg.get("ISSN", []) or []),
        "isbn": "; ".join(msg.get("ISBN", []) or []),
        "volume": msg.get("volume"),
        "issue": msg.get("issue"),
        "page": msg.get("page"),
        "url": msg.get("URL"),
        "abstract": msg.get("abstract", ""),  # puede venir en formato HTML
        "reference_count": msg.get("reference-count"),
        "is_referenced_by_count": msg.get("is-referenced-by-count"),  # “citas recibidas” en Crossref
        "subject": "; ".join(msg.get("subject", []) or []),
        "language": msg.get("language"),
        "license_url": first_or_none([lic.get("URL") for lic in msg.get("license", [])]) if msg.get("license") else None,
        "authors": join_names(msg.get("author", [])),
        "affiliations": flatten_affiliations(msg.get("author", [])),
        "year": extract_year(msg),
    }
    return record

def save_json(data: dict, path: Path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def save_csv(record: dict, path: Path):
    # Escribe una fila CSV con las claves del dict en orden estable.
    fieldnames = list(record.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(record)

def main():
    doi = sys.argv[1] if len(sys.argv) > 1 else "10.1080/23322039.2025.2560023"
    # Por buenas prácticas con Crossref, añade tu correo (opcional pero recomendado)
    mailto = None  # ejemplo: "angelo.aviles@tuuni.edu.ec"

    print(f"Consultando Crossref para DOI: {doi}")
    raw = fetch_crossref_by_doi(doi, mailto=mailto)
    record = normalize_record(raw)

    out_dir = Path("./crossref_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{record.get('doi','record').replace('/','_')}.json"
    csv_path = out_dir / "records.csv"

    save_json(raw, json_path)
    save_csv(record, csv_path)

    # Resumen en consola
    print("\n=== METADATOS PRINCIPALES ===")
    for k in ("title", "journal", "year", "authors", "doi", "url", "is_referenced_by_count"):
        print(f"{k}: {record.get(k)}")

    print(f"\nArchivos guardados:\n- JSON: {json_path}\n- CSV acumulado: {csv_path}")

if __name__ == "__main__":
    main()
