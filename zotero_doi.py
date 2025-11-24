#!/usr/bin/env python3
"""
Importa una lista de DOIs a Zotero con metadata completa desde CrossRef.

Requisitos:
    pip install pyzotero requests
"""

from pyzotero import zotero
import requests
from urllib.parse import quote

# === CONFIGURACIÓN ZOTERO ===
API_KEY = "3"  # <--- TU API KEY
USER_ID = "3"                  # <--- TU USER ID
LIBRARY_TYPE = "user"                 # "user" para biblioteca personal, "group" para grupo

# === TUS DOIS ===
DOIS = [
    "10.3790/gyil.60.1.639",
    "10.4337/jhre.2018.01.02",
    "10.1111/reel.12514",
    "10.1017/s2047102516000248",
    "10.4337/cilj.2021.01.01",
    "10.3233/epl-239011",
    "10.1093/ejil/chae071",
    "10.18196/jmh.v28i2.10988",
    "10.1163/18786561-00803006",
    "10.1007/s10784-022-09575-6",
    "10.3790/gyil.63.1.511",
    "10.1007/s10584-017-1957-5",
    "10.1163/15718107-bja10073",
    "10.1017/s2044251322000108",
    "10.1080/02646811.2016.1147887",
    "10.1515/gj-2023-0066",
    "10.4013/rechtd.2021.133.04",
    "10.1590/0034-7329201600116",
    "10.1080/10406026.2020.1718849",
    "10.4337/jhre.2021.02.04",
    "10.3390/su162310656",
    "10.1163/22116133-03301008",
    "10.54648/joia2022004",
    "10.1177/1461452919841001",
    "10.1080/02646811.2016.1120098",
    "10.1016/j.glt.2019.11.001",
    "10.3389/fmars.2024.1468210",
    "10.1080/02646811.2019.1584441",
    "10.1163/22116133-03301005",
    "10.3233/epl-239027",
    "10.1002/eet.2105",
    "10.1007/s12142-022-00674-0",
    "10.1163/18719732-12341504",
    "10.1080/1523908x.2015.1053107",
    "10.3233/epl-219002",
    "10.1080/23251042.2018.1436893",
    "10.3233/epl-219001",
]


def get_crossref_metadata(doi: str):
    """Obtiene metadata desde CrossRef para un DOI dado."""
    url = f"https://api.crossref.org/works/{quote(doi)}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] No se pudo obtener metadata de CrossRef para {doi}: {e}")
        return None

    data = resp.json()
    return data.get("message", {})


def crossref_to_zotero_item(zot, meta: dict):
    """Convierte el JSON de CrossRef en un item de tipo journalArticle para Zotero."""
    item = zot.item_template("journalArticle")

    # Título
    title_list = meta.get("title", [])
    item["title"] = title_list[0] if title_list else "Título desconocido"

    # DOI y URL
    item["DOI"] = meta.get("DOI", "")
    item["url"] = meta.get("URL", "")

    # Revista
    container_titles = meta.get("container-title", [])
    item["publicationTitle"] = container_titles[0] if container_titles else ""

    # Volumen, número, páginas
    item["volume"] = meta.get("volume", "")
    item["issue"] = meta.get("issue", "")
    item["pages"] = meta.get("page", "")

    # Fecha (issued -> date-parts)
    issued = meta.get("issued", {}).get("date-parts", [])
    if issued and isinstance(issued, list) and issued[0]:
        parts = issued[0]
        # formatea como YYYY-MM-DD o lo que haya
        date_str = "-".join(str(p) for p in parts)
        item["date"] = date_str

    # ISSN si existe
    issn_list = meta.get("ISSN", [])
    if issn_list:
        item["ISSN"] = issn_list[0]

    # Autores
    creators = []
    for author in meta.get("author", []):
        family = author.get("family", "") or ""
        given = author.get("given", "") or ""
        if not (family or given):
            continue
        creators.append({
            "creatorType": "author",
            "firstName": given,
            "lastName": family,
        })
    if creators:
        item["creators"] = creators

    return item


def exists_in_zotero(zot, doi: str) -> bool:
    """Comprueba si ya hay un ítem con ese DOI en la biblioteca."""
    try:
        results = zot.items(q=doi)
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo buscar {doi} en Zotero: {e}")
        return False

    doi_lower = doi.lower()
    for r in results:
        data = r.get("data", {})
        existing_doi = (data.get("DOI") or "").lower()
        if existing_doi == doi_lower:
            return True
    return False


def main():
    zot = zotero.Zotero(USER_ID, LIBRARY_TYPE, API_KEY)

    for doi in DOIS:
        doi = doi.strip()
        if not doi:
            continue

        print(f"\n=== Procesando DOI: {doi} ===")

        # 1. Comprobar si ya existe
        if exists_in_zotero(zot, doi):
            print(f"[SKIP] El DOI {doi} ya existe en tu biblioteca Zotero.")
            continue

        # 2. Obtener metadata desde CrossRef
        meta = get_crossref_metadata(doi)
        if not meta:
            print(f"[ERROR] No se obtuvo metadata para {doi}.")
            continue

        # 3. Convertir a item de Zotero
        item = crossref_to_zotero_item(zot, meta)

        # 4. Crear ítem en Zotero
        try:
            created = zot.create_items([item])
            keys = created.get("successful", {}).keys()
            if keys:
                print(f"[OK] {doi} añadido a Zotero. Key(s): {', '.join(keys)}")
            else:
                print(f"[ADVERTENCIA] Zotero no devolvió claves para {doi}. Respuesta: {created}")
        except Exception as e:
            print(f"[ERROR] No se pudo crear el ítem para {doi} en Zotero: {e}")


if __name__ == "__main__":
    main()
