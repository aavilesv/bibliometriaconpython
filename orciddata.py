import requests
import json

def get_orcid_metadata(orcid):
    url = f"https://pub.orcid.org/v3.0/{orcid}/record"
    headers = {"Accept": "application/json"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

data = get_orcid_metadata("0000-0001-5504-392X")

# Guardar archivo completo
with open("orcid_0000-0001-5504-392X.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Metadatos guardados.")
