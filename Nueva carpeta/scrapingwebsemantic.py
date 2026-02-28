from semanticscholar import SemanticScholar
import pandas as pd
import time
import random

# Inicializar cliente
sch = SemanticScholar(timeout=10)

queries = [
    '"communicative language competence" Spanish higher education',
    '"communicative linguistic competence" Spanish university',
    '"communicative competence" Spanish higher education',
    '"oral communicative competence" university Spanish',
    '"communicative language proficiency" Spanish university',
    '"competencia comunicativa" español educación superior',
    '"competencias lingüísticas" universidad',
    '"competencia lingüística" estudiantes universitarios'
]

resultados = []
ids_vistos = set()

print("🚀 Iniciando búsqueda con validación de duplicados...")

for query in queries:
    print(f"\n🔎 Buscando: {query}...")

    try:
        results = sch.search_paper(query, limit=30)  # más rápido

        for item in results:
            paper_id = item.paperId

            if paper_id in ids_vistos:
                continue

            ids_vistos.add(paper_id)

            resultados.append({
                "ID": paper_id,
                "Title": item.title,
                "Authors": ", ".join([a.name for a in item.authors]) if item.authors else "",
                "Year": item.year,
                "Source": item.venue,
                "Abstract": item.abstract,
                "URL": item.url,
                "Query_Origen": query
            })

        # pequeña pausa entre consultas
        time.sleep(random.uniform(2, 4))

    except Exception as e:
        print(f"Error en la consulta '{query}': {e}")
        time.sleep(5)
        continue

print("\n==============================")
print("📊 RESUMEN FINAL:")
print("Consultas realizadas:", len(queries))
print("Artículos únicos:", len(resultados))
print("==============================")

df = pd.DataFrame(resultados)
df.to_excel("resultados_unicos_validada.xlsx", index=False)

print("Archivo guardado: resultados_unicos_validada.xlsx")
