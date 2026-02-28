from scholarly import scholarly, ProxyGenerator
import pandas as pd
import time
import random

# ---------------------------------
# CONFIGURAR PROXY (seguro)
# ---------------------------------
pg = ProxyGenerator()
success = pg.FreeProxies()

if success:
    try:
        scholarly.use_proxy(pg)
        print("Proxy activado")
    except:
        print("Proxy falló, usando conexión directa")
else:
    print("Proxy no disponible, conexión directa")

# ---------------------------------
# TUS CONSULTAS
# ---------------------------------
queries = [
    '"communicative language competence" Spanish higher education', ## ya busque esto
    '"communicative linguistic competence" Spanish university',
    '"communicative competence" Spanish higher education',
    '"oral communicative competence" university Spanish',
    '"communicative language proficiency" Spanish university',
    '"competencia comunicativa" español educación superior'
]

resultados = []
max_por_query = 100

# ---------------------------------
# BÚSQUEDA
# ---------------------------------
for query in queries:
    print(f"\nBuscando: {query}")

    try:
        search_query = scholarly.search_pubs(query, patents=False, citations=False)
    except Exception as e:
        print("Error al iniciar búsqueda:", e)
        continue

    for i in range(max_por_query):
        try:
            pub = next(search_query)
            bib = pub.get("bib", {})

            resultados.append({
                "Title": bib.get("title", ""),
                "Authors": bib.get("author", ""),
                "Year": bib.get("pub_year", ""),
                "Source": bib.get("journal", ""),
                "Abstract": bib.get("abstract", ""),
                "Cited_by": pub.get("citedby", 0),
                "URL": pub.get("pub_url", ""),
                "Query": query
            })

            time.sleep(random.uniform(6, 12))

        except StopIteration:
            break
        except Exception as e:
            print("Error durante extracción:", e)
            time.sleep(random.uniform(8, 15))
            continue

# ---------------------------------
# LIMPIAR DUPLICADOS
# ---------------------------------
df = pd.DataFrame(resultados)
df = df.drop_duplicates(subset="Title")

# ---------------------------------
# GUARDAR EXCEL
# ---------------------------------
df.to_excel("articulos_linguistica_total.xlsx", index=False)

print("\nArchivo guardado con", len(df), "artículos únicos")
