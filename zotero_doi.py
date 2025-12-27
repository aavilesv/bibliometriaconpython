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
API_KEY = " TU API KEY"  # <--- TU API KEY
USER_ID = "TU USER ID"                  # <--- TU USER ID
LIBRARY_TYPE = "user"                 # "user" para biblioteca personal, "group" para grupo

# === TUS DOIS ===
DOIS = [
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
    "10.1007/s11192-009-0146-3"
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
