"""
Descarga TODO el listado de revistas de Latindex (Directorio, Catálogo 2.0 o
Descubridor de Artículos) y lo guarda en un Excel. Captura *todos* los pares
«Etiqueta: valor» que aparezcan en cada ficha, de modo que si Latindex añade
nuevos campos no hay que tocar el código.

Requisitos:
    pip install requests beautifulsoup4 pandas openpyxl
    # (opcional) pip install lxml       -> parser más rápido
"""

import requests, time, math
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://www.latindex.org/latindex"

# ---------- CONFIGURAR AQUÍ -----------------------------------------------
idModBus = 1                     # 0 = Directorio | 1 = Catálogo 2.0 | 3 = Artículos
params = {                       # Filtros (dejar '' para no filtrar)
    "pais": "",                 # p. ej. "Ecuador"
    "tema": "",                 # p. ej. "Multidisciplinarias"
    "search": ""                # texto libre
}
# --------------------------------------------------------------------------

PARSER = "html.parser"           # o "lxml" si lo tienes instalado
PAUSA  = 1                       # segundos entre páginas (cortesía)

headers = {"User-Agent": "Mozilla/5.0 (compatible; DataScraper/1.0)"}


def fetch_page(n):
    """
    Descarga la página *n* de resultados (1-based) y devuelve su HTML.
    Solo incluye en la consulta los parámetros no vacíos, y siempre
    añade idModBus y submit=Buscar.
    """
    q = {"idModBus": idModBus, "submit": "Buscar"}
    # Añade solo filtros no vacíos
    if params["pais"]:
        q["pais"] = params["pais"]
    if params["tema"]:
        q["tema"] = params["tema"]
    if params["search"]:
        q["search"] = params["search"]
    if n > 1:
        q["page"] = n

    r = requests.get(f"{BASE}/Solr/Busqueda", params=q, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def parse_html(html):
    """
    Devuelve una lista de dicts con TODOS los pares «Etiqueta: valor» que
    aparezcan en la página recibida.
    """
    soup  = BeautifulSoup(html, PARSER)
    items = []

    # Cada registro contiene un <strong>Título:</strong>
    for tit in soup.find_all("strong", string=lambda s: s.strip().startswith("Título:")):
        # Subir al div contenedor del registro
        div = tit
        while div and div.name != "div":
            div = div.parent

        datos = {}
        # Recorre todas las <strong> dentro de ese div
        for lab in div.find_all("strong"):
            etiqueta = lab.get_text(strip=True).rstrip(":")
            nxt = lab.next_sibling
            # Saltar nodos vacíos
            while nxt and (isinstance(nxt, str) and not nxt.strip()):
                nxt = nxt.next_sibling

            if nxt is None:
                valor = None
            elif isinstance(nxt, str):
                valor = nxt.strip()
            else:
                valor = nxt.get_text(" ", strip=True)

            # Concatena si la misma etiqueta se repite
            datos[etiqueta] = f"{datos.get(etiqueta,'')} | {valor}".strip(" |")

        items.append(datos)

    return items


# --------------------- DESCARGA PÁGINA 1 -----------------------------
print("Descargando página 1…")
html_1 = fetch_page(1)
rows_1 = parse_html(html_1)

if not rows_1:
    raise SystemExit("⚠️ La consulta no devolvió resultados. Revisa tus filtros.")

per_page = len(rows_1)
print(f"{per_page} registros en la primera página.")

# --------------------- RECORRER TODAS LAS PÁGINAS --------------------
all_rows = rows_1.copy()
page     = 2

while True:
    print(f"Descargando página {page}…")
    html = fetch_page(page)
    rows = parse_html(html)
    if not rows:       # deja de iterar cuando ya no hay resultados
        break
    all_rows.extend(rows)
    page += 1
    time.sleep(PAUSA)

total = len(all_rows)
print(f"\nTotal descargado: {total} registros "
      f"(≈ {math.ceil(total/per_page)} páginas).")

# --------------------- GUARDAR EN EXCEL ------------------------------
df = pd.DataFrame(all_rows)
out_file = f"latindex_resultados_idModBus{idModBus}.xlsx"
df.to_excel(out_file, index=False)
print(f"Archivo guardado: {out_file}")
