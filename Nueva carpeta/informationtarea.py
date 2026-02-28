import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 420  # cambia este valor

# Distribuciones sugeridas
age_groups = ["18-24","25-34","35-44","45-54","55+"]
age_probs  = [0.16, 0.30, 0.24, 0.20, 0.10]

gender_groups = ["Femenino","Masculino","Otro","Prefiere no responder"]
gender_probs  = [0.52, 0.45, 0.02, 0.01]

edu_groups = ["Primaria","Secundaria","Tecnica/tecnologica","Universitaria","Posgrado"]
edu_probs  = [0.10, 0.32, 0.23, 0.28, 0.07]

activity_groups = ["Microemprendimiento","Empleo dependiente","Trabajo independiente","Agricultura","Otro"]
activity_probs  = [0.46, 0.20, 0.22, 0.08, 0.04]

internet_groups = ["Siempre","Frecuente","Ocasional","Rara vez","Nunca"]
internet_probs  = [0.34, 0.33, 0.22, 0.09, 0.02]

df = pd.DataFrame({
    "F1_reside_Milagro": np.ones(N, dtype=int),
    "F2_vinculo_microfinanzas_12m": np.ones(N, dtype=int),
    "edad_rango": rng.choice(age_groups, size=N, p=age_probs),
    "genero": rng.choice(gender_groups, size=N, p=gender_probs),
    "nivel_instruccion": rng.choice(edu_groups, size=N, p=edu_probs),
    "actividad_principal": rng.choice(activity_groups, size=N, p=activity_probs),
    "acceso_internet_30d": rng.choice(internet_groups, size=N, p=internet_probs),
    "smartphone": rng.choice(["Si","No"], size=N, p=[0.90, 0.10]),
})

# Codificaciones auxiliares
internet_score = df["acceso_internet_30d"].map({"Nunca":0,"Rara vez":1,"Ocasional":2,"Frecuente":3,"Siempre":4}).astype(float)
edu_score = df["nivel_instruccion"].map({"Primaria":0,"Secundaria":1,"Tecnica/tecnologica":2,"Universitaria":3,"Posgrado":4}).astype(float)
smart_score = (df["smartphone"]=="Si").astype(float)

# Latentes (relaciones coherentes)
fintech_use = rng.normal(0, 1, N) + 0.35*(internet_score-2) + 0.25*(edu_score-2) + 0.45*(smart_score-0.85)
fintech_perc = 0.55*fintech_use + rng.normal(0, 1, N)
access = 0.30*fintech_use + 0.35*fintech_perc + rng.normal(0, 1, N)
use_services = 0.35*fintech_use + 0.25*access + rng.normal(0, 1, N)

def to_likert(x, noise=0.85):
    score = 3 + 0.75*x + rng.normal(0, noise, size=len(x))
    score = np.clip(score, 1, 5)
    return np.rint(score).astype(int)

# Items U1-U5
for col in ["U1","U2","U3","U4","U5"]:
    df[col] = to_likert(fintech_use + rng.normal(0, 0.25, N), noise=0.90)

# Items P1-P6 (P6 inverso)
df["P1"] = to_likert(fintech_perc + rng.normal(0, 0.20, N), noise=0.80)
df["P2"] = to_likert(fintech_perc + rng.normal(0, 0.20, N), noise=0.85)
df["P3"] = to_likert(fintech_perc + rng.normal(0, 0.25, N), noise=0.80)
df["P4"] = to_likert(fintech_perc + rng.normal(0, 0.20, N), noise=0.85)
df["P5"] = to_likert(fintech_perc + rng.normal(0, 0.25, N), noise=0.90)
df["P6"] = to_likert((-fintech_perc) + rng.normal(0, 0.25, N), noise=0.85)

# Items A1-A5
for col in ["A1","A2","A3","A4","A5"]:
    df[col] = to_likert(access + rng.normal(0, 0.25, N), noise=0.90)

# Items S1-S5
for col in ["S1","S2","S3","S4","S5"]:
    df[col] = to_likert(use_services + rng.normal(0, 0.25, N), noise=0.90)

# Abiertas
df["O1_barrera"] = ""
df["O2_mejora"] = ""

# Alfa de Cronbach
def cronbach_alpha(data: pd.DataFrame) -> float:
    k = data.shape[1]
    item_vars = data.var(ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    return (k/(k-1)) * (1 - item_vars.sum()/total_var)

# Inversion de P6 para analisis
df["P6_rev"] = df["P6"].map({1:5,2:4,3:3,4:2,5:1})

alpha_u = cronbach_alpha(df[["U1","U2","U3","U4","U5"]])
alpha_p = cronbach_alpha(df[["P1","P2","P3","P4","P5","P6_rev"]])
alpha_a = cronbach_alpha(df[["A1","A2","A3","A4","A5"]])
alpha_s = cronbach_alpha(df[["S1","S2","S3","S4","S5"]])

print("Alpha U:", alpha_u)
print("Alpha P:", alpha_p)
print("Alpha A:", alpha_a)
print("Alpha S:", alpha_s)

df.to_csv("datos_simulados_fintech_milagro.csv", index=False, encoding="utf-8")
