# -*- coding: utf-8 -*-
"""
02 - Lematización y canonización de Author/Index Keywords (IN-PLACE)
- Lematización spaCy
- UK→US / sinónimos
- Canonización de frases/tokens (VPN, IoT, QoS; neural networks→neural network)
- Filtro final
- Log de cambios (antes/después)
"""

import re, time
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import pandas as pd
from unidecode import unidecode
import spacy

# ====== Rutas ======
INPUT_CSV   = r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replace.csv"
OUTPUT_CSV  =r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1replacelematizar.csv"
CHANGE_LOG  = r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\02_lemmatize_canonize_log.csv"
KW_COLS     = ["Author Keywords", "Index Keywords"]

# ====== Config ======
EXCEPTION_PHRASES = {
    # mismas excepciones que en el script 01
"data",    "e-leadership", "digital leadership", "virtual leadership", "transformational leadership",
    "industry 4 0", "digital transformation", "higher education", "covid-19", "artificial intelligence",
    "machine learning", "bibliometric analysis", "systematic review", "knowledge management",
    "pls-sem", "organizational performance", "leadership theory", "leadership style",
    "innovation", "education", "performance", "trust", "motivation", "resilience", "organization", "management",
    "e-leadership", "digital leadership", "virtual leadership", "technology-enhanced learning",
    "higher education institutions (heis)", "tertiary education", "student leadership",
    "technology acceptance model", "leader–member exchange (lmx)", "path goal theory",
    "structuration theory", "critical race theory", "actor-network theory",
    "fuzzy set qualitative comparative analysis", "partial least squares", "PLS-SEM",
    "structural equation modeling", "latent dirichlet allocation (lda) analysis",
    "k-means clustering", "science mapping", "bibliometric historiography", "scimat",
    "internet of things (iot)", "cloud computing", "cybersecurity", "service oriented architecture (soa)",
    "enterprise resource planning", "enterprise social networks",
    "covid-19", "covid-19 pandemic", "severe acute respiratory syndrome coronavirus 2",
    "telemedicine", "telehealth", "telenursing",
      "e-leadership", "(e)-leadership", "(un)ethical e-leadership",
    "technology-enhanced learning",
    "virtual communities of practice", "virtual professional development (vpd)",
    "vuca environment",
    "3d printing", "3d web and virtual worlds", "3d/4d/5d printing",
    "4ir", "4.0 era", "industry 4.0", "industry 5.0",
    "5g mobile communication systems",
    "internet of things (iot)", "cloud computing", "cybersecurity",
    "service oriented architecture (soa)", "enterprise resource planning",
    "computer-mediated communication (cmc)",
    "technology acceptance model",
    "leader–member exchange (lmx)", "lmx",
    "partial least squares", "pls", "pls-sem", "smartpls",
    "structural equation model", "structural equation modeling", "structural equation models",
    "latent dirichlet allocation (lda) analysis",
    "analytical hierarchy process (ahp)",
    "fuzzy set qualitative comparative analysis", "fsqca",
    "co-occurrence analysis", "co-word analysis",
    "science mapping", "citespace", "scimat", "bibliometric mapping", "bibliometrics",
    "covid-19", "covid-19 pandemic", "covid-19 vaccines", "covid-19 post covid-19",
    "severe acute respiratory syndrome coronavirus 2", "coronavirus disease 2019",
    "telemedicine", "telehealth", "telenursing",
    "strategic alignment", "strategic agility", "strategic flexibility",
    "absorptive capacity", "ambidextrous leadership", "ambidextrous learning", "ambidextrous innovation",
    "work design", "work organization", "work team",
    "thought leadership", "responsible leadership", "visionary leadership",
      # critical / cross-cultural
    "critical digital citizenship", "critical discourse analysis", "critical realism",
    "critical reflections", "critical success factors (csf)", "critical failure factors (cffs)",
    "cross-cultural communication", "cross-cultural differences", "cross-cultural leadership",
    "cross-cultural negotiation", "cross-cultural organization", "cross-cultural organizations",
    "cross-cultural scenario", "cross-cultural teams", "cross-cultural validation",
    "cross-level moderation model", "cross-organisational collaboration",

    # customer / CRM / lógicas
    "customer-dominant logic", "customer-service", "customer journey",
    "customer experience orientation", "customer relationship management",

    # cyber / CPS / ética-ley
    "cyber-physical systems (cps)", "cyber ethics", "cyber law", "cyber risks",
    "cyberbullying", "cyberloafing", "cybernetics",

    # data / métodos
    "data-driven approach", "data-driven culture", "data-driven education",
    "data visualization", "data envelopment analysis", "data lake", "data repository", "datafication",

    # deep / ML
    "deep neural networks", "deep q-learning", "deep echo state network (desn)",

    # teorías/modelos/gobernanza
    "deliberative democracy", "diffusion of innovations", "delone and mclean model",
    "conservation of resources (cor) theory", "actor-network theory", "attention-based view",
    "attention-based views", "complexity leadership theory",

    # digital leadership / transformation (familias)
    "digital leadership framework", "digital leadership scale",
    "digital leadership competency", "digital leadership competences",
    "digital leadership capabilities", "digital leadership skill",
    "digital leadership skills development", "digital leadership and culture",
    "digital leadership in marketing", "digital leadership system",
    "digital transformation (dt)", "digital transformation capabilities",
    "digital transformation leadership", "digital transformation in higher education",
    "digital transformation initiative", "digital transformation projects",
    "digital twin", "digital twins", "digital sovereignty",
    "digital platform ecosystems", "digital entrepreneurial ecosystems",
    "digital health applications", "digital health strategies",
    "digital health technology", "digital health transformation",

    # edu
    "education 4 0", "digital learning environment", "digital learning environments",
    "digital learning platforms", "distance learning leadership",
    "distance education and online learning",

    # e- prefijo
    "e-leadership behavior", "e-leadership competencies", "e-leadership practices",
    "e-leadership roles", "e-learning platforms", "e-government performance",
    "e-portfolio analysis", "e-service quality", "e-signature", "e-sports",
    "e-work environment", "e-work self-efficacy",

    # varios identitarios
    "dark side of leadership", "design science research", "design science",
    "design research", "destination management", "destructive leadership",
    "distributed leadership work", "district leadership", "enterprise resource planning",
    "enterprise social media", "enterprise governance of it", "epistemic network analysis",
    "evidence-based policy", "evidence-based practices", "equity-centered leadership",
    "evolutionary algorithms", "evolutionary game", "evolutionary games",
    "extended-baldridge performance indicator", "facebook communities", "facebook groups",
    "vuca environment", "bani",
     # Marcos/teorías/modelos/actas
    "federal information security management act (fisma)",
    "ferpa",
    "fiedler’s contingency model",
    "full range leadership model",
    "five forces model",
    "grounded theory approach",
    "grounded practical theory",
    "hofstede’s cultural dimensions",
    "heliotropic leadership",
    "health-oriented leadership",
    "industrial-organizational psychology",
    "job demands–resources (jd-r) model",
    "information systems success model",
    "fuzzy set qualitative comparative analysis (fsqca)",
    "fsqca3 0software",
    "grey wolf optimizer",
    "floyd algorithm",

    # Familia digital/AI que funcionan como etiquetas
    "generative artificial intelligence",
    "generative ai",
    "green digital transformation",
    "green digital transformational leadership",
    "green digital mindset",
    "global virtual team leadership",

    # Educación superior / trabajo híbrido (etiquetas)
    "higher education institution",
    "higher education pedagogy",
    "higher education online learning",
    "higher education performance",
    "hybrid work model",
    "hybrid workplace",
    "hybrid work teams",

    # ICT / IS / IT (formas canónicas)
    "information and communication technologies (ict)",
    "information technology (it)",
    "is/it strategic planning",

    # Otros nombres/entidades
    "federal government",
    "g20",
    "fridays for future",
    "greta thunberg",
    "hillary clinton",
    "harvard medical school",
    "imperial college london",
    "hong kong",
    "ireland",
    "google",
    "github",
    "ikea",
       # Países/estados/ciudades/zonas
    "Jordanian manufacturing sector","Karnataka","Kerala","Kentucky","Kuwait",
    "Korea","Korea and China","Latin America","London","New York City",
    "Melbourne  Australia","Hong Kong",
    # Niveles educativos / ámbitos
    "K-12","K-12 schools","K-12 international schooling",
    # Leyes/estándares/agencias
    "FERPA","FISMA","OECD","NHS",
    # Modelos/métodos/teorías/métricas
    "Latent Dirichlet Allocation","LDA","Latent growth curve modeling",
    "Latent profile analysis","Latent transition analysis",
    "Kirkpatrick model","McKinsey’s 7S","Lyapunov stability","Lyapunov function method",
    "Newton–Raphson method","Model predictive control","MIMO systems",
    "OKR (Objectives and Key Results)","fsQCA","MCDM","LIWC","Leximancer",
    # Algoritmos / optimizadores
    "k-means++ clustering","Grey wolf optimizer",
    # Teorías/constructos clásicos
    "Hofstede’s cultural dimensions","leader–member exchange (LMX)","LMX","LMXSC",
    # Plataformas/marcas/tecnologías
    "GitHub","LinkedIn","MOOCs",
    # Expresiones establecidas
    "global virtual team leadership","learning management systems","LMS",
    "knowledge-intensive business services","massively multiplayer online games (MMOGs)","MMOGs",
       # Genéricos frecuentes (mantener frase, lematizar tokens)
    "journalism innovation","juvenile",
    "key characteristics","key competencies","key dimensions","key factors","key issues",
    "key performance indicators","key topics",
    "knowledge creation","knowledge sharing behavior","knowledge translation",
    "labor productivity","lagged effects",
    "learning strategies","learning styles","learning processes","learning outcomes",
    "learning experience design","learning environments","learning organization",
    "library services","library management",
    "life satisfaction","life skills",
    "logistics companies","logistics leadership",
    "longitudinal analysis","longitudinal studies",
    "management capabilities","management practices","management skills","management strategies",
    "market competition","market demand","marketing communications","marketing performance",
    "mental models","mentoring programs",
    "mixed-methods study","mobile working",
    "motivation system","motivating language",
    "multilevel analysis","multilingual support",
    "network leadership","network effects",
    "nonlinear programming",
    "occupational health and safety",
    "online learning leadership","online discussions","online program leadership",
    "online participation","online platforms",
    "organizational learning","organizational performance",
    "knowledge workers","lean manufacturing","learning management",   # Modalidades/recursos abiertos y siglas establecidas
    "Open Educational Resources (OER)",
    "Open Distance Digital Education (ODDE)",
    "Open Science",
    "Open Access",
    "Open University",
    "Open Knowledge Maps",
    # Aprendizaje/formatos específicos
    "Peer-Led Team Learning (PLTL)",
    "Project-Based Learning",
    "Problem-Based Learning",
    # Algoritmos/modelos/métodos
    "Q-learning",
    "Particle Swarm Optimization (PSO)",
    "PLS-ANN",
    "Partial Least Squares (PLS)",
    "PLS regression",
    "RFID (radio frequency identification)",
    "Reinforcement Learning",
    "Lyapunov stability",  # por si no lo tenías aún
    # Ecosistemas/plataformas
    "Platform ecosystem",
    "Platform ecosystems",
    # Indicadores/ODS
    "SDG 3",
      # Online / abierto
    "online project-based learning","online public services","online purchase intention",
    "online questionnaire","online teamwork","online training program","open coding",
    "open collaborative innovation","open education","open practices",
    # Operaciones / desempeño
    "operational efficiency","operational performance","operational standards",
    "operations strategy","organizational performance","organizational transformation",
    "organizational assessment","organizational support","organizational justice",
    "organizational inertia","organizational learning culture",
    # Liderazgo / percepciones
    "open leadership","paternalistic leadership","path-goal leadership",
    "people-oriented leadership","perceived organizational politics",
    "perceived transformational leadership","relational leadership",
    # Recursos/persona–organización
    "person–organization (P–O) fit","person–job fit","resource-based view",
    # Educación / escuela
    "school digitalization","school effectiveness","school management",
    "school technology leadership","project leadership","project performance",
    # Salud / paciente
    "patient-centered care","patient engagement","patient experience",
    # Políticas / público
    "public governance","public management","public service motivation",
    # Métricas / calidad
    "quality assurance","quality culture","quality of education",
    "quality of working life","performance management systems","performance measurement",
    # Datos / IA
    "predictive analytics","recommender systems","process mining",
    # Trabajo remoto
    "remote work performance","remote leadership traits","remote school leadership",
    # Intención/compra/comportamiento
    "purchase intention","prosocial behavior","opinion sharing",
      # Teorías / modelos canónicos
    "Self-Determination Theory",
    "Self-regulated learning (SRL)",
    "Situational Leadership Theory",
    "Servant Leadership",
    "Society 5.0",
    "Socio-technical systems (STS)",
    "Support Vector Machines (SVM)",
    "Structural Equation Modeling (SEM)",
    "Theory of Planned Behavior (TPB)",
    "UTAUT (Unified Theory of Acceptance and Use of Technology)",
    "UTAUT3",
    "TPACK",
    "TQM",
    "Triple Bottom Line (TBL)",
    "Transactive Memory",
    "Swift trust",
    "Strategy-as-practice",
    "Small and Medium-sized Enterprises (SMEs)",
    "Unmanned Aerial Vehicles (UAV)",
    # Instrumentos/servqual
      "self-control","self-management","self-organization","self-paced learning",
    "self-regulated learning skills","self-determination","self-engagement",
    "self-leadership skills","self-sacrifice","self-transcendent value",
    # Sentido/meaning
    "sense-making methodology","sense making","sense of power","sense of purpose",
    # Diseños/metodologías
    "semiotics","semantic analysis","sequence analysis","sensitivity analysis",
    "serial mediation","simulation model","simulation training",
    # Liderazgo/servicio
    "service leadership","service climate","service innovation performance",
    "shared control","shared mental models","shared leadership",
    # Social/organizacional
    "sharing economy","social influence","social innovation","social identity continuity",
    "social intelligence","social presence theory","social participation",
    "social sustainability","social networks","social connectedness",
    # Educación/estudiantes/equipos
    "situated learning","small-group learning","student learning outcomes",
    "student satisfaction","student leadership training","team building",
    "team cohesion","team coordination","team knowledge sharing",
    "team performance management","team satisfaction","teamwork skills",
    # Cadena de suministro
    "supply chain agility","supply chain collaboration","supply chain performance",
    # Estrategia/gestión
    "strategic communication","strategic decision making","strategic alignment theory",
    "strategic partnership","strategic technologies","system integration",
    # Trabajo/estrés
    "surface acting","stress assessment","staff commitment","staff selection",
    # Tecnología/aceptación
    "technology governance","technology implementation","technology management",
    "technology-driven work arrangements","topic modeling","text-mining",
     "Virtual Learning Environment (VLE)",
    "Virtual Professional Development (VPD)",
    "Virtual Professional Leadership Learning Communities (VPLCs)",
    "Virtual Reality (VR)",
    "Visual SLAM",
    "Visual Analytics",
    "VUCA",
    "Zero Trust",
    "Warp-PLS",
    "WHO-5 Well-Being Index",
    "Work-from-home (WFH)",
    "Word of Mouth (WOM)",
     "virtual instructional leadership","virtual management","virtual mentoring",
    "virtual mentoring and coaching","virtual meetings","virtual team efficiency",
    "virtual team interactions","virtual team management","virtual team productivity",
    "virtual team strategies for diversity and inclusion","virtual working",
    "virtual workspace","virtual training","virtual transition",
    "virtual professional development",

    # Educación/PD
    "virtual global classroom","virtual learning policy considerations",
    "virtual schooling","virtual school","vocational education training",
    "work-based learning","workshop","writing workshop",

    # Trabajo/organizacional
    "work arrangements","work autonomy","work flexibility","work motivation",
    "work meaningfulness","workplace innovation","workplace mental health",
    "workplace technologies","workplace transformation","workplace well-being",
    "workload stress","workplace diversity","workplace deviance",
    "work–life interface","work schedule","work practices",

    # Comportamiento/voz/visibilidad
    "voice behavior","visibility","volatility",

    # Analítica/visualización
    "visual analytics","visual communication","visual network analysis",
    "visual risk communication",

    # Seguridad / confianza
    "zero trust",

    # Métodos/otros
    "virtual simulation environments","visual surveillance systems",
      "visualization", "voice", "youth", "workload", "workshop", "world",
    "ability", "abilities", "adoption", "alignment", "allocation",
    "analysis", "assessment", "advantage", "advocacy",
    "algorithm", "analytics", "application", "appropriation", "architecture",
    "arrangement", "barrier", "benefit", "belonging",
    "behavior", "commitment", "communication", "community", "competition",
    "compliance", "complexity", "confidence", "configuration", "conflict",
    "connection", "cooperation", "coordination", "coping", "cost",
    "creativity", "crisis", "decision-making", "design", "development",
    "diffusion", "effectiveness", "efficiency", "engagement", "environment",
    "evaluation", "evidence", "experience", "exploration", "facilitation",
    "feedback", "flexibility", "forecasting", "framework",
    "goal", "governance", "growth", "guidance", "identity", "inclusion",
    "influence", "innovation", "integration", "intention", "interaction",
    "investment", "knowledge", "leadership", "learning",
    "management", "measurement", "methodology", "model", "monitoring",
    "motivation", "network", "outcome", "performance", "perception",
    "planning", "policy making", "practice", "preparedness", "prevention",
    "privacy", "problem solving", "process", "productivity",
    "professional development", "professional practice", "public policy",
    "quality", "readiness", "resilience", "resource", "responsibility",
    "retention", "risk", "satisfaction", "security", "strategy", "stress",
    "support", "sustainability", "taxonomy", "teamwork", "training",
    "trust", "uncertainty", "validation", "value", "vision", "work",
       "criticism", "crossover", "crowdsourcing", "culinary",
    "cultural aspects", "cultural changes", "cultural constraints",
    "cultural identity", "cultural issue", "cultural settings",
    "curriculum alignment", "curriculum guidelines", "curriculum resource",
    "customer acceptance", "customer behavior", "customer involvement",
    "decision makers", "decision outcomes", "decision making process",
    "data acquisition", "data extraction", "data gathering", "data management",
    "data processing", "data protection", "data visualization",
    "decentralized control", "decentralized decision-making",
    "definition", "determinants", "development programs", "development status",
    "dialogue", "differentiation", "diffusion process", "dimensions",
    "discrimination", "disengagement", "dispersion", "disinformation",
    "disasters", "distribution channel", "domains", "dominance",
    "drivers", "dynamic patterns",
      # Feedback / governance / finanzas / firma
    "feedback acceptance", "feedback quality", "feedback system",
    "financial governance", "financial impacts", "financial performance",
    "firm culture", "firm resilience", "firm strategy", "firm size",

    # Trabajo, flexibilidad, grupos
    "flexible working system", "flexible working systems",
    "group dynamics", "group decision making", "group collaboration processes",
    "guideline adherence", "good practices",

    # Gobierno y gobernanza
    "governance models", "governance structures",
    "government agencies", "government intervention", "government projects",

    # Educación / atributos
    "graduate attributes", "formative assessment", "formal education",

    # Innovación / estrategia
    "innovation adoption", "innovation capacity", "innovation diffusion",
    "innovation strategy", "innovation system",
    "innovative culture", "innovative pedagogy", "innovative practices",

    # Identidad / inclusión
    "identity formation", "identity work",
    "inclusion", "inclusiveness", "inclusivity",

    # Información / gestión
    "information analysis", "information gathering", "information leadership",
    "information management", "information security management",

    # Institucional
    "institutional culture", "institutional development",
    "institutional framework", "institutional pressure",

    # Integración
    "integrated approach", "integrated design framework",
    "integration group",

    # Interpersonal / internacional / internet / entrevistas
    "interpersonal relationships", "interpersonal trust",
    "internationalization", "internet use", "interviews",

    # Trabajo y desempeño
    "job engagement", "job performance", "job transformation",
    "job satisfaction", "job security", "job tension",

    # Salud en el trabajo (genérico, no clínico)
    "health at work",

    # Liderazgo genérico
    "group leadership", "great leaders",


}

EXCEPTION_NOUNS = {"autism spectrum disorder"}

BRIT_US = {
    "behaviour": "behavior", "behaviours": "behavior",
    "organisation": "organization", "organisations": "organizations",
}

PHRASE_CANON = [
   
]

TOKEN_CANON = {

}

# ====== spaCy ======
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ====== Utils ======
RE_SINGLE = re.compile(r'^[a-z]$')
RE_DIGITS = re.compile(r'^\d+$')
_PH_MARK  = "\uFFF1"

EXC_PATTERNS = [(re.compile(rf"\b{re.escape(p)}\b"), p) for p in sorted(EXCEPTION_PHRASES, key=len, reverse=True)]

def limpieza_basica(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unidecode(s.lower())
    s = re.sub(r"[^a-z0-9'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mask_exception_phrases(text: str):
    mapping, idx = {}, 0
    for pat, phrase in EXC_PATTERNS:
        key = f"{_PH_MARK}{idx}"
        if pat.search(text):
            mapping[key] = phrase
            text = pat.sub(key, text); idx += 1
    return text, mapping

def unmask_exception_phrases(text: str, mapping: dict):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def aplicar_mapa(texto: str, mapping: dict) -> str:
    for k, v in mapping.items():
        texto = re.sub(rf"\b{re.escape(k)}\b", v, texto)
    return texto

def canonize_phrases(text: str) -> str:
    for k, v in PHRASE_CANON:
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text

def canonize_tokens(text: str) -> str:
    toks = text.split()
    return " ".join([TOKEN_CANON.get(t, t) for t in toks])

def filtrar_basura(term: str) -> str:
    toks = []
    for t in term.split():
        if RE_SINGLE.match(t):  continue
        if RE_DIGITS.match(t):  continue
        toks.append(t)
    return " ".join(toks).strip()

def lemmatize_spacy(texto: str) -> str:
    doc = nlp(texto)
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct: continue
        orig, lem = tok.text, tok.lemma_
        if tok.tag_ == "VBG":
            out.append(orig)
        elif tok.pos_ == "NOUN" and orig in EXCEPTION_NOUNS:
            out.append(orig)
        else:
            out.append(lem)
    return " ".join(out).strip()

@lru_cache(maxsize=200_000)
def normalize_single_keyword(kw: str) -> str:
    kw = limpieza_basica(kw)
    if not kw: return ""

    # 1) No tocamos ortografía aquí (ya corregida en 01). Protegemos solo frases-excepción.
    kw, exc_map = mask_exception_phrases(kw)

    # 2) Lematizar
    kw = lemmatize_spacy(kw)

    # 3) UK→US / sinónimos
    kw = aplicar_mapa(kw, BRIT_US)

    # 4) Canonización
    kw = canonize_phrases(kw)
    kw = canonize_tokens(kw)

    # 5) Restaurar excepciones
    kw = unmask_exception_phrases(kw, exc_map)

    # 6) Filtro final
    kw = filtrar_basura(kw)
    return kw

def normalize_cell(cell: str) -> str:
    if not isinstance(cell, str):
        return ""
    raw = [t.strip() for t in cell.split(';') if t.strip()]
    out = []
    for term in raw:
        norm = normalize_single_keyword(term)
        if norm:
            out.append(norm)   # ← sin 'seen'
    return "; ".join(out)

# ====== MAIN ======
if __name__ == "__main__":
    t0 = time.perf_counter(); ts0 = datetime.now()
    df = pd.read_csv(INPUT_CSV).fillna("")
    Path(CHANGE_LOG).parent.mkdir(parents=True, exist_ok=True)

    log_rows = []
    print("Antes (recuento únicos):")
    for c in KW_COLS:
        if c in df.columns:
            nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
            print(f"  {c}: {nuniq}")

    for col in KW_COLS:
        if col not in df.columns: continue
        print(f"\nLematizando/canonizando: {col} ...")
        before = df[col].astype(str)
        after  = before.apply(normalize_cell)
        changed_mask = (before != after)
        if changed_mask.any():
            tmp = pd.DataFrame({
                "row_index": df.index[changed_mask],
                "column": col,
                "before": before[changed_mask].values,
                "after":  after[changed_mask].values
            })
            log_rows.append(tmp)
        df[col] = after

    print("\nDespués (recuento únicos):")
    for c in KW_COLS:
        if c in df.columns:
            nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
            print(f"  {c}: {nuniq}")

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    if log_rows:
        pd.concat(log_rows, ignore_index=True).to_csv(CHANGE_LOG, index=False, encoding="utf-8")
        print(f"\n📝 Log de cambios: {CHANGE_LOG}")
    else:
        print("\n📝 Sin cambios registrados.")

    t1 = time.perf_counter(); ts1 = datetime.now()
    print("\n" + "="*60)
    print("📅 Inicio:", ts0.strftime("%Y-%m-%d %H:%M:%S"))
    print("🕒 Fin   :", ts1.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"⏱️ Tiempo total de ejecución: {t1 - t0:.2f} s")
    print("📁 CSV final:", OUTPUT_CSV)
    print("="*60)
