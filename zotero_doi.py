#!/usr/bin/env python3
"""
Importa una lista de DOIs a Zotero con metadata completa (incluyendo Abstract) desde CrossRef.
"""
#pip install pyzotero==1.7.5 requests==2.32.3
#C:/Users/User/AppData/Local/Programs/Python/Python39/python.exe -m pip install pyzotero==1.7.5 requests==2.32.3

from pyzotero import zotero
import requests
from urllib.parse import quote
import re  # <--- NUEVO: Para limpiar las etiquetas HTML del abstract

# === CONFIGURACIÓN ZOTERO ===
# ¡IMPORTANTE!: Regenera tu API Key en Zotero, la anterior fue expuesta.
API_KEY = "URysNjB8GswoQDUjM4I6Kd9p" 
USER_ID = "10438425"                  
LIBRARY_TYPE = "user"                 

# === TUS DOIS ===
DOIS = [
"10.4067/S0717-95022024000501239",
    "10.3390/en17164083",
    "10.1016/j.prime.2023.100348",
    "10.3390/su15097440",
    "10.3390/s22228980",
    "10.1016/j.ohx.2022.e00355",
    "10.14483/23448393.18861",
    "10.1109/ACCESS.2022.3144044",
    "10.3390/app10134606",
    "10.1109/ACCESS.2018.2876197",
    "10.1186/s40064-016-3250-x",
    "10.1016/j.rmclc.2022.11.003"
]

def clean_html_tags(text):
    """Elimina etiquetas tipo <jats:p> o <i> del texto usando expresiones regulares."""
    if not text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

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

    # === NUEVO: Procesamiento del Abstract ===
    raw_abstract = meta.get("abstract", "")
    # Limpiamos las etiquetas XML/HTML que suele mandar CrossRef
    item["abstractNote"] = clean_html_tags(raw_abstract)
    # ========================================

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
        date_str = "-".join(str(p) for p in parts)
        item["date"] = date_str

    # ISSN
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
        # Se usa q=doi para búsqueda rápida
        results = zot.items(q=doi, limit=1) 
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

    print(f"Procesando {len(DOIS)} DOIs...")

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

        # 3. Convertir a item de Zotero (Ahora incluye Abstract)
        item = crossref_to_zotero_item(zot, meta)

        # 4. Crear ítem en Zotero
        try:
            created = zot.create_items([item])
            keys = created.get("successful", {}).keys()
            if keys:
                print(f"[OK] {doi} añadido a Zotero con Abstract. Key(s): {', '.join(keys)}")
            else:
                print(f"[ADVERTENCIA] Zotero no devolvió claves para {doi}. Respuesta: {created}")
        except Exception as e:
            print(f"[ERROR] No se pudo crear el ítem para {doi} en Zotero: {e}")

if __name__ == "__main__":
    main()