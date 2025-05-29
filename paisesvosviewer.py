import pandas as pd
import spacy
from rapidfuzz import process, fuzz
import re

# —————————————————————————————
# 0) Tu lista canónica de países (la misma que ya tienes)
# —————————————————————————————
countries = [
    "Algeria", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Canada", "China",
    "Colombia", "Costa Rica", "Czech Republic", "Denmark", "Egypt", "Chile", 
    "Ethiopia", "Finland", "France", "Germany", "Ghana",
    "Greece", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Lebanon",
    "Lithuania", "Malaysia", "Mexico", "Morocco", "Nepal", "Netherlands",
    "New Zealand", "Nigeria", "Norway", "Oman", "Pakistan", "Palestine",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
    "Russia", "Rwanda", "Saudi Arabia", "Senegal", "Serbia", "Singapore",
    "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka",
    "Sweden", "Switzerland", "Taiwan", "Thailand", "Tunisia", "Turkey",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Uzbekistan", "Vietnam", "Zimbabwe", "Ivory Coast", "Kuwait",
    "Croatia", "Afghanistan", "Albania", "Andorra", "Angola", "Armenia",
    "Azerbaijan", "Bahamas", "Bahrain", "Barbados", "Belarus", "Belize",
    "Benin", "Bhutan", "Bolivia", "Botswana", "Brunei Darussalam",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon",
    "Central African Republic", "Chad", "Comoros", "Congo", "Cuba",
    "Cyprus", "Djibouti", "Dominican Republic", "Ecuador", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Fiji", "Gabon",
    "Gambia", "Georgia", "Grenada", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Iceland", "Jamaica", "Kiribati", "Kyrgyzstan", "Laos",
    "Latvia", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Luxembourg",
    "Madagascar", "Malawi", "Maldives", "Mali", "Malta", "Marshall Islands",
    "Mauritania", "Mauritius", "Micronesia", "Monaco", "Mongolia",
    "Montenegro", "Mozambique", "Myanmar", "Namibia", "Nauru", "Niger",
    "North Macedonia", "Palau", "Panama", "Papua New Guinea", "Paraguay",
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Seychelles", "Sierra Leone", "Slovakia",
    "Solomon Islands", "Somalia", "South Sudan", "Sudan", "Suriname",
    "Tajikistan", "Tanzania", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Turkmenistan", "Tuvalu", "Uganda", "Uruguay",
    "Vanuatu", "Venezuela", "Yemen", "Zambia", "Swaziland"
]

# —————————————————————————————
# 1) Diccionario de sinónimos para fallback (usa las mismas claves del array)
# —————————————————————————————
synonyms_map = {c.lower(): c for c in countries}

# Patrón regex que busca cualquier país canónico en Minúsculas
pattern = re.compile(r'\b(' + '|'.join(map(re.escape, synonyms_map.keys())) + r')\b', flags=re.IGNORECASE)

# spaCy y rapidfuzz
nlp = spacy.load("en_core_web_sm")

def extract_countries_ner(text: str) -> str:
    seen = []
    out  = []
    doc  = nlp(text)
    # (A) Intento con spaCy + fuzzy
    for ent in doc.ents:
        if ent.label_ == "GPE":
            cand = ent.text.strip()
            best, score, _ = process.extractOne(cand, countries, scorer=fuzz.token_set_ratio)
            #token_set_ratio, Esta función es ideal cuando las cadenas contienen las mismas palabras pero pueden
            #tener palabras adicionales o faltantes, y el orden no es relevante.


            if score >= 95 and best not in seen:
                seen.append(best)
                out.append(best)
    # (B) Fallback regex: captura cualquier país no cogido por fuzzy
    for m in pattern.finditer(text):
        canon = synonyms_map[m.group(1).lower()]
        if canon not in seen:
            seen.append(canon)
            out.append(canon)
    return "; ".join(out)

# —————————————————————————————
# 2) Carga y aplicación
# —————————————————————————————
ruta_entrada = r"G:\\Mi unidad\\Master en administración y empresas\\articulo 3\\data\\datawos_scopus.csv"
ruta_salida  = r"G:\\Mi unidad\\Master en administración y empresas\\articulo 3\\data\\datawos_scopus_con_paises.csv"

df = pd.read_csv(ruta_entrada, encoding="utf-8-sig")
df["Texto_combinado"] = df["Affiliations"].fillna("") + " " + df["Authors with affiliations"].fillna("")
df["Countries"] = df["Texto_combinado"].apply(extract_countries_ner)

# 1) Asigna el contenido de Countries a Affiliations, y al mismo tiempo elimina Countries
df['Affiliations'] = df.pop('Countries')
df["Authors with affiliations"] = df["Affiliations"]
# 2) Elimina Texto_combinado
df.drop(columns=['Texto_combinado'], inplace=True)
# Guardar
df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

print("✅ Proceso completo: spaCy+fuzzy + fallback regex (solo pocos valores faltantes incluidos).")
