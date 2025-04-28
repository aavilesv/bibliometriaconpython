import pandas as pd
import spacy
countries = [    "Algeria",    "Argentina",     "Australia",     "Austria",     "Bangladesh",     "Belgium",   
               "Bosnia and Herzegovina",     "Brazil",    "Bulgaria",     "Canada",     "China",     "Colombia",     
               "Costa Rica",     "Czech Republic",     "Denmark",     "Egypt",     "Ethiopia",     "Finland",     "France",   
 "Germany",     "Ghana",     "Greece",     "Hungary",     "India",     "Indonesia",     "Iran",     "Iraq",    
 "Ireland",     "Israel",     "Italy",     "Japan",     "Jordan",     "Kazakhstan",     "Kenya",     "Lebanon",   
 "Lithuania",     "Malaysia",     "Mexico",     "Morocco",     "Nepal",     "Netherlands",     "New Zealand",
 "Nigeria",     "Norway",     "Oman",     "Pakistan",     "Palestine",     "Peru",     "Philippines", 
 "Poland",     "Portugal",     "Qatar",     "Republic of Korea",     "Romania",   
 "Russia",     "Rwanda",     "Saudi Arabia",     "Senegal",     "Serbia",     
 "Singapore",     "Slovenia",     "South Africa",     "South Korea",     "Spain",     "Sri Lanka",     "Sweden",     
 "Switzerland",     "Taiwan",     "Thailand",     "Tunisia",     "Turkey",     "Ukraine",     "United Arab Emirates",    
 "United Kingdom",     "United States",     "Uzbekistan",     "Vietnam",     "Zimbabwe", ]
# ——————————————
# 1) Rutas de entrada y salida
# ——————————————
ruta_entrada = r"G:\Mi unidad\2025\master kevin castillo\artículo nuevo\data\datawos_scopus.csv"
ruta_salida  = r"G:\Mi unidad\2025\master kevin castillo\artículo nuevo\data\datawos_scopus_con_paises.csv"

# ——————————————
# 2) Cargar datos y modelo spaCy
# ——————————————
df = pd.read_csv(ruta_entrada, encoding="utf-8-sig")
nlp = spacy.load("en_core_web_sm")

# ——————————————
# 3) Función que extrae GPE/LOC únicos y en orden
# ——————————————
def extract_countries_ner(text):
    seen = set()
    out = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE"):
            name = ent.text.strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return "; ".join(out)

# ——————————————
# 4) Combinar ambas columnas en un solo texto
# ——————————————
df["Texto_combinado"] = (
    df["Affiliations"].fillna("") + " " + 
    df["Authors with affiliations"].fillna("")
)

# ——————————————
# 5) Extraer países y guardar resultados
# ——————————————
df["Countries"] = df["Texto_combinado"].apply(extract_countries_ner)

# Si quieres también un archivo “explotado” con una fila por país:
df_exploded = df.assign(
    Country = df["Countries"].str.split("; ")
).explode("Country")

df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
df_exploded.to_csv(
    r"G:\Mi unidad\2025\master kevin castillo\artículo nuevo\data\datawos_scopus_con_paises_explode.csv",
    index=False, encoding="utf-8-sig"
)

print(f"✅ Listo: archivos guardados en:\n • {ruta_salida}\n • datawos_scopus_con_paises_explode.csv")
