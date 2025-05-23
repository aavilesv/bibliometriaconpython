import pandas as pd
import re

# 1) Leer CSV
ruta = r"G:\\Mi unidad\\2025\\Master Estefania Landires\\datawos_scopus.csv"
df = pd.read_csv(ruta).fillna("")

# 2) Lista de países y diccionario de sinónimos
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


synonyms_map = {c.lower(): c for c in countries}



# 3) Compilar patrones
country_vals = '|'.join(map(re.escape, synonyms_map.values()))
# Detecta comas mal puestas tras un país, independientemente de siguiente palabra
sep_pattern = re.compile(rf'({country_vals})\s*,\s*(?=[A-Z])')
# Patron de contenido entre corchetes
bracket_pattern = re.compile(r'\[.*?\]')

def normalize_cell(raw: str) -> str:
    # 1) Eliminar contenido entre corchetes
    text = bracket_pattern.sub('', raw)
    # 2) Reemplazar comas tras país por ';'
    text = sep_pattern.sub(r'\1; ', text)
    # 3) Split en fragments
    frags = [frag.strip() for frag in text.split(';') if frag.strip()]
    # 4) Asegurar país al final de cada fragmento
    norm = []
    for i, frag in enumerate(frags):
        # Si termina en un país canónico, OK
        if any(frag.endswith(c) for c in synonyms_map.values()):
            norm.append(frag)
        else:
            # Intentar tomar país de siguiente fragmento
            if i + 1 < len(frags):
                for key, val in synonyms_map.items():
                    if frags[i+1].lower().startswith(val.lower()):
                        norm.append(f"{frag}, {val}")
                        break
                else:
                    norm.append(frag)
            else:
                norm.append(frag)
    # 5) Unir de nuevo
    return '; '.join(norm)

# 4) Aplicar a columnas
df['Affiliations'] = df['Affiliations'].apply(normalize_cell)
df['Authors with affiliations'] = df['Authors with affiliations'].apply(normalize_cell)
# 9) Guarda el resultado
out = r"G:\\Mi unidad\\2025\\Master Estefania Landires\\datawos_scopusfinal.csv"

df.to_csv(out, index=False)
print("Resultado guardado en:", out)


