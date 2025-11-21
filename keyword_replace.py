import pandas as pd
import re

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv(r"G:\Mi unidad\2025\Master  FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopus.csv")
KW_COLS     = ["Author Keywords", "Index Keywords"]
# Diccionario de palabras clave a reemplazar: clave = palabra a buscar (en minúsculas), valor = palabra de reemplazo
palabras_clave_reemplazo = {
    # Clima
    "climate-change": "climate change",
    "global warming": "climate change",
    "climate change policy": "climate policy",
    "climate-change adaptation": "climate change adaptation",
    "climate-change mitigation": "climate change mitigation",

    # Gobernanza ambiental / Derecho
    "environmental-regulation": "environmental regulation",
    "environmental regulations": "environmental regulation",
    "environmental-policy": "environmental policy",
    "environmental governance": "environmental governance",
    "global environmental governance": "global environmental governance",
    "international environmental law": "international environmental law",
    "international law": "international law",
    "environmental law": "environmental law",
    "environmental legislation": "environmental legislation",
    "environmental justice": "environmental justice",
    "climate justice": "climate justice",

    # Gobernanza multinivel
    "multi-level governance": "multilevel governance",
    "multi level governance": "multilevel governance",
    "network governance": "network governance",
    "collaborative governance": "collaborative governance",
    "adaptive governance": "adaptive governance",
    "co-management": "co management",
    "comanagement": "co management",

    # Tratados / organismos
    "unfccc": "united nations framework convention on climate change",
    "united-nations framework convention on climate change": "united nations framework convention on climate change",
    "redd": "redd+",
    "redd plus": "redd+",
    "redd+": "redd+",
    "paris agreement": "paris agreement",
    "kyoto protocol": "kyoto protocol",

    # Emisiones / carbono
    "co2": "co2 emissions",
    "co2 emission": "co2 emissions",
    "co2 emissions": "co2 emissions",
    "carbon emission": "carbon emissions",
    "carbon emissions": "carbon emissions",
    "carbon-dioxide emissions": "carbon dioxide emissions",
    "carbon dioxide": "carbon dioxide emissions",
    "carbon dioxide emission": "carbon dioxide emissions",
    "greenhouse gas": "greenhouse gas emissions",
    "greenhouse-gas emissions": "greenhouse gas emissions",
    "ghg emissions": "greenhouse gas emissions",

    # Transición energética
    "energy-consumption": "energy consumption",
    "energy consumption": "energy consumption",
    "energy efficiency": "energy efficiency",
    "energy-efficiency": "energy efficiency",
    "renewable energy": "renewable energy",
    "energy transition": "energy transition",

    # Desarrollo sostenible
    "sustainable development": "sustainable development",
    "sustainable-development": "sustainable development",
    "sdgs": "sustainable development goals",
    "sustainable development goals": "sustainable development goals",
    "green economy": "green economy",
    "green-growth": "green growth",
    "green growth": "green growth",
    "eco-innovation": "eco innovation",
    "environmental kuznets curve": "environmental kuznets curve",

    # Territorio
    "land-use": "land use",
    "land use change": "land use change",
    "land-use change": "land use change",

    # Otros compuestos frecuentes
    "air-pollution": "air pollution",
    "air pollution": "air pollution",
    "air-quality": "air quality",
    "panel-data": "panel data",
    "foreign direct-investment": "foreign direct investment",
    "foreign direct investment": "foreign direct investment",
    "european-union": "european union",
    "united-states": "united states",
    "research-and-development": "research and development",
    "technological-innovation": "technological innovation",
    "social-ecological systems": "social ecological systems",
    "public-participation": "public participation",
    "social network analysis": "social network analysis",

    # participación / stakeholders
    "policy making": "policy making",
    "policy-making": "policy making",
    "public participation": "public participation",
    "stakeholder participation": "stakeholder participation",
    "stakeholder analysis": "stakeholder analysis",
    "community participation": "community participation",

    # clima / políticas climáticas
    "climate-change policy": "climate policy",
    "climate change governance": "climate governance",
    "change adaptation": "climate change adaptation",
    "change mitigation": "climate change mitigation",
    "climate-change mitigation": "climate change mitigation",
    "climate-change impacts": "climate change impacts",
    "climate-induced migration": "climate migration",
    "cambio climático": "climate change",
    "climate effect": "climate change impacts",
    "climate resilience": "climate resilience",
    "climate finance": "climate finance",
    "climate litigation": "climate litigation",
    "climate risk": "climate risk",
    "climate justice": "climate justice",

    # gobernanza / derecho
    "environmental policies": "environmental policy",
    "environmental treaties": "environmental treaties",
    "forest governance": "forest governance",
    "ocean governance": "ocean governance",
    "urban governance": "urban governance",
    "agri-environmental governance": "agri-environmental governance",
    "environmental governance and regulation": "environmental governance and regulation",
    "polycentric governance": "polycentric governance",
    "earth system governance": "earth system governance",
    "administrative-law": "administrative law",
    "administrative law": "administrative law",
    "international environmental agreements": "international environmental agreements",
    "trade agreements": "trade agreements",
    "soft law": "soft law",
    "rights of nature": "rights of nature",
    "human-rights": "human rights",
    "indigenous rights": "indigenous rights",
    "indigenous knowledge": "indigenous knowledge",
    "traditional knowledge": "traditional knowledge",
    "convention on biological diversity": "convention on biological diversity",
    "environmental security": "environmental security",
    "environmental tax": "environmental tax",

    # emisiones / carbono
    "carbon-dioxide": "carbon dioxide emissions",
    "carbon dioxide emissions": "carbon dioxide emissions",
    "carbon dioxide": "carbon dioxide emissions",
    "carbon emission intensity": "carbon intensity",
    "carbon emissions trading": "carbon emissions trading",
    "carbon emission mitigation": "carbon emission mitigation",
    "carbon emission reduction": "carbon emission reduction",
    "carbon market": "carbon market",
    "carbon markets": "carbon market",
    "carbon pricing": "carbon pricing",
    "carbon taxes": "carbon tax",
    "carbon intensity": "carbon intensity",
    "low-carbon": "low carbon",
    "low-emission zones": "low emission zones",
    "gas emissions": "greenhouse gas emissions",
    "greenhouse gases": "greenhouse gas emissions",
    "greenhouse gas emissions": "greenhouse gas emissions",

    # energía
    "electricity-generation": "electricity generation",
    "electricity generation": "electricity generation",
    "electricity consumption": "electricity consumption",
    "energy consumption": "energy consumption",
    "energy intensity": "energy intensity",
    "energy poverty": "energy poverty",
    "energy security": "energy security",
    "wind energy": "wind energy",
    "natural-gas": "natural gas",
    "oil and gas": "oil and gas",

    # economía verde / desarrollo
    "circular economy": "circular economy",
    "green development": "green development",
    "green infrastructure": "green infrastructure",
    "green spaces": "green spaces",
    "green total factor productivity": "green total factor productivity",
    "green total factor energy efficiency": "green total factor energy efficiency",
    "green technology": "green technology",
    "green credit": "green credit",
    "green credit policy": "green credit policy",
    "green supply chain": "green supply chain",

    # métodos / modelos
    "decomposition analysis": "decomposition analysis",
    "difference-in-difference": "difference in differences",
    "difference-in-differences model": "difference in differences",
    "did": "difference in differences",
    "time-series": "time series",
    "panel-data": "panel data",
    "panel data": "panel data",
    "spatial durbin model": "spatial durbin model",
    "q-methodology": "q methodology",
    "quantile regression": "quantile regression",
    "regression": "regression",
    "gravity model": "gravity model",
    "dea": "data envelopment analysis",
    "error-correction": "error correction model",
    "cross-sectional dependence": "cross sectional dependence",

    # participación / redes / sociedad
    "social media": "social media",
    "social networks": "social network",
    "social network": "social network",
    "social justice": "social justice",
    "civil-society": "civil society",
    "civil society": "civil society",
    "ngos": "ngos",

    # geografía / países / regiones
    "brics": "brics",
    "uk": "united kingdom",
    "us": "united states",
    "eu": "european union",
    "south africa": "south africa",
    "global south": "global south",
    "developing-countries": "developing countries",
    "developing country": "developing countries",
    "emerging economies": "emerging economies",
    "oecd countries": "oecd countries",
    "latin america": "latin america",
    "amazonia": "amazonia",
    "amazon": "amazon",
    "british-columbia": "british columbia",
    "baltic sea": "baltic sea",
    "chile": "chile",
    "peru": "peru",
    "ecuador": "ecuador",
    "vietnam": "vietnam",
    "papua new guinea": "papua new guinea",
    "pacific islands": "pacific islands",

    # urbano / territorio
    "urban-development": "urban development",
    "urban development": "urban development",
    "urban area": "urban areas",
    "urban politics": "urban politics",
    "urban planning": "urban planning",

    # otros relevantes
    "environmental-management": "environmental management",
    "environmental management": "environmental management",
    "environmental activism": "environmental activism",
    "environmental concern": "environmental concern",
    "environmental diplomacy": "environmental diplomacy",
    "environmental flows": "environmental flows",
    "environmental migrants": "environmental migrants",
    "environmental technology": "environmental technology",
    "environmental assessment": "environmental assessment",
    "environmental impact assessment": "environmental impact assessment",
    "ecological footprint": "ecological footprint",
    "ecosystem-based management": "ecosystem based management",
    "ecological restoration": "ecological restoration",
    "nature-based solutions": "nature based solutions",
    "nature conservation": "nature conservation",
    "forest carbon": "forest carbon",
    "forest conservation": "forest conservation",
    "fisheries management": "fisheries management",
    "flood governance": "flood governance",
    "flood risk": "flood risk",
    "flooding": "flooding",
    "groundwater": "groundwater",
    "water-quality": "water quality",
    "water quality": "water quality",
    "water-resources": "water resources",
    "water resources": "water resources",
    "hydropower": "hydropower",
    "wetlands": "wetlands",

    # instituciones / análisis institucional
    "institutional analysis": "institutional analysis",
    "institutional change": "institutional change",
    "institutional design": "institutional design",
    "institutional theory": "institutional theory",
    "informal regulation": "informal regulation",
    "local-government": "local government",
    "local government": "local government",
    "fiscal decentralization": "fiscal decentralization",
    "devolution": "devolution",

    # finanzas / empresa
    "foreign direct investment": "foreign direct investment",
    "foreign direct-investment": "foreign direct investment",
    "financing constraints": "financing constraints",
    "financialization": "financialization",
    "green credit policy": "green credit policy",
    "cost-benefit-analysis": "cost benefit analysis",
    "cost of debt": "cost of debt",
    "esg": "esg",
    "firm value": "firm value",

    # otros normalizables
    "british-columbia": "british columbia",
    "built environment": "built environment",
    "big data": "big data",
    "ai": "artificial intelligence",
    "twitter": "twitter",
    "social network": "social network",
    "socio-ecological systems": "social ecological systems",
    "rescaling environmental governance": "rescaling environmental governance",
    "environmental impact assessment": "environmental impact assessment",
    "environmental economics": "environmental economics",
    "environmental technology": "environmental technology",
    "geoengineering": "geoengineering",
      # Gobernanza / derecho internacional
    "international environmental-law": "international environmental law",
    "international environmental governance": "international environmental governance",
    "international governance": "international governance",
    "procedural justice": "procedural justice",
    "private environmental governance": "private environmental governance",
    "policy approach": "policy approach",
    "policy uncertainty": "policy uncertainty",
    "precautionary approach": "precautionary approach",
    "meta-governance": "meta governance",
    "polycentric governance": "polycentric governance",
    "marine governance": "marine governance",
    "multilateral environmental agreements": "multilateral environmental agreements",
    "world polity": "world polity",
    "derechos humanos": "human rights",
    "self-regulation": "self regulation",
    "soft law": "soft law",
    "duty of care": "duty of care",

    # Tratados y regímenes climáticos
    "montreal protocol": "montreal protocol",
    "stockholm convention": "stockholm convention",
    "nationally determined contributions": "nationally determined contributions",
    "common but differentiated responsibilities": "common but differentiated responsibilities",
    "common concern of humankind": "common concern of humankind",
    "climate regime": "climate regime",
    "unep": "unep",
    "agenda 2030": "agenda 2030",

    # Cambio climático (sinónimos)
    "climate-change policy": "climate policy",
    "change policy": "climate policy",
    "change mitigation": "climate change mitigation",
    "change adaptation": "climate change adaptation",
    "climate change adaptation and mitigation": "climate change adaptation and mitigation",
    "climate change communication": "climate change communication",
    "climate displacement": "climate migration",
    "climate migration": "climate migration",
    "climate variability": "climate variability",
    "climate science": "climate science",
    "climate forcing": "climate forcing",
    "climate change impacts": "climate change impacts",

    # Gobernanza oceánica / marina
    "marine environment": "marine environment",
    "marine fisheries": "marine fisheries",
    "ocean": "ocean",
    "coastal governance": "coastal governance",

    # Emisiones / carbono
    "carbon emission performance": "carbon emissions performance",
    "carbon efficiency": "carbon efficiency",
    "carbon emission trading pilot": "carbon emission trading pilot",
    "carbon emission trading": "carbon emissions trading",
    "carbon mitigation": "carbon mitigation",
    "carbon peak": "carbon peak",
    "carbon performance": "carbon performance",
    "carbon price": "carbon pricing",
    "carbon productivity": "carbon productivity",
    "co 2 emissions": "co2 emissions",
    "co2 emissions evidence": "co2 emissions",
    "co2 mitigation": "co2 mitigation",

    # Energía / transición
    "alternative energy": "alternative energy",
    "energy justice": "energy justice",
    "energy poverty": "energy poverty",
    "energy transitions": "energy transition",
    "energy transition": "energy transition",

    # Biodiversidad y ecosistemas
    "agroforestry": "agroforestry",
    "biodiversity governance": "biodiversity governance",
    "biodiversity loss": "biodiversity loss",
    "wildlife": "wildlife",
    "coral-reefs": "coral reefs",
    "forestry": "forestry",

    # Instituciones / teoría institucional
    "advocacy coalition framework": "advocacy coalition framework",
    "discursive institutionalism": "discursive institutionalism",
    "institutional analysis": "institutional analysis",
    "institutional theory": "institutional theory",
    "institutional design": "institutional design",
    "differential treatment": "differential treatment",
    "resource-management": "resource management",

    # Economía / sostenibilidad / desarrollo
    "low-carbon development": "low carbon development",
    "low-carbon pilot policy": "low carbon pilot policy",
    "kuznets curve hypothesis": "environmental kuznets curve",
    "decoupling economic-growth": "decoupling economic growth",
    "circular economy": "circular economy",
    "biofuel sustainability": "biofuel sustainability",
    "financial constraints": "financial constraints",
    "digital finance": "digital finance",
    "digital transformation": "digital transformation",
    "fossil fuel": "fossil fuels",
    "fossil fuels": "fossil fuels",

    # Participación / sociedad
    "public-policy": "public policy",
    "public perceptions": "public perceptions",
    "public policies": "public policies",
    "public support": "public support",
    "social-responsibility": "social responsibility",
    "social movements": "social movements",
    "community engagement": "community engagement",

    # Territorio
    "loess plateau": "loess plateau",
    "qinghai-tibet plateau": "qinghai tibet plateau",
    "philippines": "philippines",
    "russia": "russia",
    "southeast asia": "southeast asia",
    "netherlands": "netherlands",
    "central-asia": "central asia",

    # Métodos
    "life-cycle assessment": "life cycle assessment",
    "psm-did": "difference in differences",
    "stirpat": "stirpat model",
    "meta-governance": "meta governance",
    "content analysis": "content analysis",
    "text analysis": "textual analysis",

    # Justicia / derechos
    "right to a healthy environment": "right to a healthy environment",
    "social vulnerability": "social vulnerability",
    "procedural justice": "procedural justice",
      # clima, justicia, gobernanza
    "global environmental-change": "global environmental change",
    "global environmental assessments": "global environmental assessment",
    "gobernanza ambiental global": "global environmental governance",
    "governance gaps": "governance gaps",
    "hybrid governance": "hybrid governance",
    "international climate change law": "international climate change law",
    "international-law": "international law",
    "international agreements": "international agreements",
    "international treaties": "international treaties",
    "transnational climate governance": "transnational climate governance",
    "transnational environmental governance": "transnational environmental governance",
    "planetary boundaries": "planetary boundaries",
    "planetary justice": "planetary justice",
    "intergenerational equity": "intergenerational equity",
    "intergenerational justice": "intergenerational justice",
    "procedural justice": "procedural justice",
    "reparative justice": "reparative justice",

    # tratados, organismos, regímenes
    "intergovernmental panel on climate change": "ipcc",
    "kyoto": "kyoto protocol",
    "united-nations": "united nations",
    "united nations framework convention on climate change (unfccc)":        "united nations framework convention on climate change",
    "unccd": "unccd",

    # cambio climático / urbano
    "local climate action": "local climate action",
    "local climate governance": "local climate governance",
    "urban-politics": "urban politics",
    "urban politics": "urban politics",
    "urban environmental governance": "urban environmental governance",
    "urban experimentation": "urban experimentation",
    "urban heat island": "urban heat island",
    "urban resilience": "urban resilience",
    "urban water management": "urban water management",
    "urban wetlands": "urban wetlands",
    "urban political ecology": "urban political ecology",

    # emisiones / carbono / energía
    "fuel poverty": "energy poverty",
    "green economic efficiency": "green economic efficiency",
    "green transition": "green transition",
    "green transformation": "green transformation",
    "green bonds": "green bonds",
    "green building": "green building",
    "green consumption": "green consumption",
    "green investment": "green investment",
    "green new deal": "green new deal",
    "green patent": "green patent",
    "green production performance": "green production performance",
    "green transformation": "green transformation",
    "green transition": "green transition",
    "geothermal energy": "geothermal energy",
    "renewable energy-consumption": "renewable energy consumption",
    "renewable energy consumption": "renewable energy consumption",
    "sustainable energy": "sustainable energy",

    # low carbon / economía
    "low-carbon city": "low carbon city",
    "low-carbon economy": "low carbon economy",
    "low carbon economy": "low carbon economy",
    "low-carbon transitions": "low carbon transitions",
    "low-carbon development": "low carbon development",

    # gobernanza regional / subnacional
    "regional environmental governance": "regional environmental governance",
    "regional governance": "regional governance",
    "subnational diplomacy": "subnational diplomacy",
    "subnational governments": "subnational governments",

    # participación, sociedad, stakeholders
    "grassroots": "grassroots",
    "participatory governance": "participatory governance",
    "participatory processes": "participatory processes",
    "participatory action research": "participatory action research",
    "political-participation": "political participation",
    "political participation": "political participation",
    "social movement": "social movement",
    "social movements": "social movements",
    "social norms": "social norms",
    "social license": "social license",
    "stakeholder theory": "stakeholder theory",
    "stakeholders": "stakeholders",
    "local communities": "local communities",
    "local governance": "local governance",
    "local participation": "local participation",

    # pueblos indígenas / conocimiento
    "indigenous people": "indigenous peoples",
    "indigenous": "indigenous peoples",
    "indigeneity": "indigeneity",
    "inuit": "inuit",
    "traditional ecological knowledge": "traditional ecological knowledge",
    "traditional knowledge": "traditional knowledge",
    "situated knowledges": "situated knowledges",

    # gobernanza empresarial / responsabilidad
    "green bonds": "green bonds",
    "green building": "green building",
    "organizational behavior and the environment":
        "organizational behaviour and the environment",
    "private governance": "private governance",
    "social-responsibility": "social responsibility",
    "pro-environmental behavior": "pro environmental behavior",
    "nonfinancial disclosure": "nonfinancial disclosure",
    "public environmental concern": "public environmental concern",

    # enfoques teóricos / ciencia y sociedad
    "science and technology studies": "science and technology studies",
    "science diplomacy": "science diplomacy",
    "transdisciplinary science": "transdisciplinary science",
    "transformative change": "transformative change",
    "sustainability transformations": "sustainability transformations",
    "sustainability governance": "sustainability governance",
    "sustainable governance": "sustainable governance",
    "sustainability fix": "sustainability fix",

    # acuerdos comerciales, cadenas globales
    "global supply chain": "global supply chains",
    "global value chain": "global value chains",
    "global value chains": "global value chains",
    "multilateral trade": "multilateral trade",

    # regiones / países
    "g20": "g20",
    "g7": "g7",
    "hong kong": "hong kong",
    "ireland": "ireland",
    "japan": "japan",
    "mexico": "mexico",
    "nepal": "nepal",
    "new-zealand": "new zealand",
    "new zealand": "new zealand",
    "pakistan": "pakistan",
    "peruvian amazonia": "peruvian amazonia",
    "south asia": "south asia",
    "turkey": "turkey",
    "vietnamese mekong delta": "mekong delta",

    # instrumentos de política / regulación
    "market-based environmental regulation": "market based environmental regulation",
    "policy coherence": "policy coherence",
    "policy diffusion": "policy diffusion",
    "policy entrepreneurs": "policy entrepreneurs",
    "policy evaluation": "policy evaluation",
    "policy innovation": "policy innovation",
    "policy instruments": "policy instruments",
    "policy platform": "policy platform",
    "policymaking": "policy making",
    "monetary-policy": "monetary policy",
    "taxation": "taxation",
    "emission taxes": "emission taxes",

    # justicia, género, interseccionalidad
    "gender diversity": "gender diversity",
    "intersectionality": "intersectionality",
    "planetary justice": "planetary justice",
    "reparative justice": "reparative justice",

    # herramientas, software, indicadores
    "vosviewer": "vosviewer",
    "gis": "gis",

    # biodiversidad, bosques, costas
    "marine environmental protection": "marine environmental protection",
    "marine litter": "marine litter",
    "marine protected areas": "marine protected areas",
    "tropical forest": "tropical forests",
    "wetland": "wetlands",
    "wetlands": "wetlands",
    "wildfire": "wildfire",
    "wildlife": "wildlife",

    # otros relevantes
    "sids": "small island developing states",
    "small island developing states": "small island developing states",
    "small islands": "small islands",
    "smart cities": "smart cities",
    "smart city": "smart cities",
    "waste management": "waste management",
    "water security": "water security",
    "watershed": "watershed",
        "world trade organization": "wto",
    "g20": "g20",
    "g7": "g7",
    "green belt and road initiative": "belt and road green initiative",
    "belt and road initiative (bri)": "belt and road initiative",
    "belt and road initiative": "belt and road initiative",
    "green bonds": "green bonds",
    "green transformation": "green transformation",
    "green transition": "green transition",
    "hybrid governance": "hybrid governance",
    "informal institutions": "informal institutions",
    "international legal framework": "international legal framework",
    "international agreements": "international agreements",
    "international organization": "international organization",
    "international climate change law": "international climate change law",
    "international human rights law": "international human rights law",
    "marine environmental protection": "marine environmental protection",
    "marine governance": "marine governance",
    "participatory governance": "participatory governance",
    "participatory processes": "participatory processes",
    "policy diffusion": "policy diffusion",
    "policy innovation": "policy innovation",
    "policy instruments": "policy instruments",
    "policy evaluation": "policy evaluation",
    "policy coherence": "policy coherence",
    "policymaking": "policy making",
    "political-participation": "political participation",
    "political feasibility": "political feasibility",
    "procedural justice": "procedural justice",
    "public-policy": "public policy",
    "public-opinion": "public opinion",
    "public health": "public health",
    "sustainability governance": "sustainability governance",
    "sustainability transformations": "sustainability transformations",
    "sustainable governance": "sustainable governance",
    "transboundary pollution": "transboundary pollution",
    "transboundary cooperation": "transboundary cooperation",
    "transnational governance": "transnational governance",
    "transnational environmental governance": "transnational environmental governance",
    "united nations": "united nations",
    "wto": "wto",
    "unfccc": "unfccc",
    "unccd": "unccd",
    "uk": "united kingdom",
    "oecd": "oecd",
    "smart cities": "smart cities",
    "smart city": "smart cities",
    "renewable energy-consumption": "renewable energy consumption",
    "renewable energy consumption": "renewable energy consumption",
    "risk-management": "risk management",
    "risk governance": "risk governance",
    "rights of nature": "rights of nature",
    "rule of law": "rule of law",
    "regional governance": "regional governance",
    "regional cooperation": "regional cooperation",
    "small island developing states": "small island developing states",
    "sids": "small island developing states",
    "urban environmental governance": "urban environmental governance",
    "urban resilience": "urban resilience",
    "urban water management": "urban water management",
    "water security": "water security",
    "waste management": "waste management",
       # Gobernanza y derecho ambiental/internacional
    "conference des parties": "conference of the parties",
    "conference of parties": "conference of the parties",
    "conference of the parties": "conference of the parties",
    "cop 26": "cop26",
    "cop agenda": "cop agenda",
    "cop negotiations": "cop negotiations",
    "copenhagen accord": "copenhagen accord",
    "copenhagen agreement": "copenhagen agreement",
    "convention on biological diversity (cbd)": "convention on biological diversity",
    "convention on biological diversity": "convention on biological diversity",
    "convention on the rights of the child": "convention on the rights of the child",
    "convención sobre los derechos del niño": "convention on the rights of the child",
    "convention-cadre des nations unies sur les changements climatiques":
        "united nations framework convention on climate change",
    "cooperative federalism": "cooperative federalism",
    "constitutional rights": "constitutional rights",
    "constitutionalism": "constitutionalism",
    "constitutionalization of environment": "constitutionalization of environment",
    "control of corruption": "control of corruption",
    "courts": "courts",
    "cross-border governance": "cross-border governance",
    "digital environmental governance": "digital environmental governance",
    "disaster risk governance": "disaster risk governance",
    "distributive and procedural justice": "distributive and procedural justice",
    "due diligence": "due diligence",
    "due process": "due process",
    "derechos supraindividuales ambientales": "diffuse environmental rights",

    # Corporativo / clima / gobernanza empresarial
    "corporate environmental governance": "corporate environmental governance",
    "corporate environmental practices": "corporate environmental practices",
    "corporate climate change disclosure": "corporate climate change disclosure",
    "corporate pollution": "corporate pollution",
    "corporate pollution emissions": "corporate pollution emissions",
    "corporate social responsibility (csr)": "corporate social responsibility",
    "corporate sustainability reporting": "corporate sustainability reporting",

    # Convenios / modelos de gobernanza ecológica relevantes
    "ecosystem-based adaptation": "ecosystem-based adaptation",
    "ecosystem-based adaptation (eba)": "ecosystem-based adaptation",
    "ecosystem-based fisheries management": "ecosystem-based fisheries management",
    "ecological law": "ecological law",
    "ecological governance": "ecological governance",
    "economic governance": "economic governance",
    "economic policy uncertainty": "economic policy uncertainty",
    "earth system justice": "earth system justice",

    # Instrumentos de mercado / regulación climática
    "emission trading scheme": "emissions trading scheme",
    "emission trading scheme (ets)": "emissions trading scheme",
    "emissions trading schemes": "emissions trading scheme",
    "emission tax": "emission tax",
    "emission reduction policy": "emission reduction policy",
    "emission reduction commitments": "emission reduction commitments",

    # Países / regiones de interés
    "costa rica": "costa rica",
    "cote d'ivoire": "cote d'ivoire",
    "congo basin": "congo basin",
    "congo basin forests": "congo basin forests",

    # Modelos de participación / justicia
    "conflict-management": "conflict management",
    "participatory governance": "participatory governance",
    "disaster risk-management": "disaster risk management",
    # ENERGÍA
    "energy-conservation": "energy conservation",
    "energy-saving policy": "energy saving policy",
    "energy carbon emission efficiency": "energy carbon emission efficiency",
    "energy conservation and emission reduction": "energy conservation and emission reduction",
    "energy ecological efficiency": "energy ecological efficiency",
    "energy governance": "energy governance",
    "energy innovations": "energy innovations",
    "energy management": "energy management",
    "energy policies": "energy policies",
    "energy policy network": "energy policy network",
    "energy politics": "energy politics",
    "energy productivity": "energy productivity",
    "energy resource": "energy resources",
    "energy resources": "energy resources",
    "energy saving and emission reduction": "energy saving and emission reduction",
    "energy saving and emission reduction plan": "energy saving and emission reduction plan",
    "energy savings and climate change mitigation": "energy savings and climate change mitigation",
    "energy structure transformation": "energy structure transformation",
    "energy sustainability": "energy sustainability",
    "energy taxes": "energy taxes",
    "energy technologies": "energy technologies",
    "energy technology innovation": "energy technology innovation",
    "energy transition minerals": "energy transition minerals",
    "energy affordability": "energy affordability",

    # DERECHO / GOBERNANZA AMBIENTAL
    "enviromental law": "environmental law",
    "environment governance": "environmental governance",
    "environment protection": "environmental protection",
    "environment quality": "environmental quality",
    "environment regulations": "environmental regulation",
    "environment conservation": "environmental conservation",
    "environmental-education": "environmental education",
    "environmental-impact assessment": "environmental impact assessment",
    "environmental-quality": "environmental quality",
    "environmental treaty law": "environmental treaty law",
    "environmental regulation (er)": "environmental regulation",
    "environmental regulation competition": "environmental regulation competition",
    "environmental regulation policy": "environmental regulation policy",
    "environmental regulation of companies": "environmental regulation of companies",
    "environmental protection tax": "environmental protection tax",
    "environmental taxes": "environmental taxes",
    "environmental taxation": "environmental taxation",
    "environmental taxes and subsidies": "environmental taxes and subsidies",

    # ESG / RESPONSABILIDAD / FINANZAS SOSTENIBLES
    "environmental  social  and governance": "environmental social governance",
    "environmental social and governance": "environmental social governance",
    "environmental social governance (esg)": "environmental social governance",
    "esg investing": "esg investing",
    "esg scores": "esg scores",

    # GOBERNANZA / POLÍTICA AMBIENTAL
    "environment governance": "environmental governance",
    "environmental co-governance system": "environmental co-governance system",
    "environmental commons": "environmental commons",
    "environmental foreign policy": "environmental foreign policy",
    "environmental governance agreements": "environmental governance agreements",
    "environmental governance and policy": "environmental governance and policy",
    "environmental governance lessons": "environmental governance lessons",
    "environmental governance networks": "environmental governance networks",
    "environmental governance policy": "environmental governance policy",
    "environmental governmentality": "environmental governmentality",
    "environmental paradiplomacy": "environmental paradiplomacy",
    "environmental regional governance": "environmental regional governance",
    "environmental regimes": "environmental regimes",
    "environmental resource management": "environmental resource management",
    "environmental responsibility": "environmental responsibility",
    "environmental science and policy": "environmental science and policy",

    # OTROS TÉRMINOS AMBIENTALES RELEVANTES
    "environmental kuznets's curve": "environmental kuznets curve",
    "environmental kuznets curves": "environmental kuznets curve",
    "environmental ngos": "environmental ngos",
    "environmental nongovernmental organizations": "environmental ngos",
    "environmental footprints": "environmental footprints",
    "environmental protest": "environmental protest",
    "environmental risk perception": "environmental risk perception",
    "environmental vulnerability": "environmental vulnerability",
    "environmental resilience": "environmental resilience",
    "environmental movement": "environmental movement",
    "environmental movements": "environmental movements",

    # EUROPA / EU ETS / POLÍTICA EUROPEA
    "eu-ets": "eu ets",
    "eu emissions trading scheme": "eu emissions trading scheme",
    "eu emissions trading system": "eu emissions trading system",
    "eu environmental acquis": "eu environmental acquis",
    "eu environmental policy": "eu environmental policy",
    "eu external policy": "eu external policy",
    "eu habitats directive": "eu habitats directive",

    # GLOBAL / GOBERNANZA GLOBAL
    "global environmental challenges": "global environmental challenges",
    "global environmental policy": "global environmental policy",
    "global environmental problems": "global environmental problems",
    "global health governance": "global health governance",
    "global justice": "global justice",
    "global law": "global law",
    "global north-south relations": "global north-south relations",
    "global stocktake": "global stocktake",
    "global sustainability": "global sustainability",

    # GOBERNANZA (ESPAÑOL)
    "gobernanza ambiental": "environmental governance",
    "gobernanza climatica": "climate governance",
    "gobernanza urbana": "urban governance",
    "gobierno local": "local government",

    # GREEN / ECONOMÍA VERDE Y TRANSICIÓN
    "green finance policy": "green finance policy",
    "green financial development": "green financial development",
    "green financial system": "green financial system",
    "green financing": "green financing",
    "green growth": "green growth",
    "green economic growth": "green economic growth",
    "green economic recovery": "green economic recovery",
    "green energy": "green energy",
    "green energy consumption": "green energy consumption",
    "green hydrogen": "green hydrogen",
    "green infrastructure (gi)": "green infrastructure",
    "green infrastructure": "green infrastructure",
    "green jobs": "green jobs",
    "green manufacturing development": "green manufacturing development",
    "green organizational identity": "green organizational identity",
    "green space": "green space",
    "green spaces": "green spaces",
    "greenspace": "green space",
    "green supply chain management": "green supply chain management",
    "green tax": "green tax",
    "green technologies": "green technologies",
    "green technology progress": "green technology progress",
    "greenhouse-gas": "greenhouse gas emissions",
    "greenhouse-gas disclosure": "greenhouse gas disclosure",
    "greenhouse gas (ghg)": "greenhouse gas emissions",
    "greenhouse gas abatement": "greenhouse gas abatement",
    "greenhouse gas emission": "greenhouse gas emissions",
    "greenhouse gas mitigation": "greenhouse gas mitigation",
    "greenhouse gas targets": "greenhouse gas targets",

    # OTROS RELEVANTES
    "forestland governance": "forestland governance",
    "forestry ecological efficiency": "forestry ecological efficiency",
    "fracking policy": "fracking policy",
    "global activism": "global activism",
    "global adaptation governance": "global adaptation governance",
    "global administrative law": "global administrative law",
    "global environmental commons": "global environmental commons",
    "global forest policy": "global forest policy",
    "global health law": "global health law",
    "global obligations": "global obligations",
    "global pact": "global pact",
    "goal-based governance": "goal-based governance",
    "governance and management": "governance and management",
    "governance capacity": "governance capacity",
    "governance ecosystem": "governance ecosystem",
    "governing by targets": "governing by targets",
    "government environmental regulation": "government environmental regulation",
    "government environmental regulations": "government environmental regulations",
    "government environmental concern": "government environmental concern",
    "government environmental governance information disclosure":
        "government environmental governance information disclosure",

            # clima urbano / salud
    "heat-island": "urban heat island",
    "heat island impact": "urban heat island impact",
    "health-risk assessment": "health risk assessment",
    "health risk": "health risk",

    # relaciones sociedad–ambiente
    "human-environment": "human environment relations",
    "human-environment relations": "human environment relations",
    "human-geography": "human geography",
    "human environment": "human environment",
    "human and non-human beings": "human and non-human beings",

    # derechos humanos y clima/ambiente
    "human rights and climate change": "human rights and climate change",
    "human rights and the environment": "human rights and the environment",
    "human rights beyond borders": "human rights beyond borders",
    "human rights covenant": "human rights covenant",
    "human rights law": "human rights law",

    # gobernanza híbrida
    "hybrid environmental governance": "hybrid environmental governance",

    # pueblos indígenas
    "indigenous governance": "indigenous governance",
    "indigenous lands": "indigenous lands",
    "indigenous or aboriginal peoples": "indigenous peoples",
    "indigenous organizations": "indigenous organizations",
    "indigenous peoples? rights": "indigenous peoples rights",
    "indigenous peoples’ rights": "indigenous peoples rights",
    "indigenous population": "indigenous population",

    # energía / emisiones industriales relevantes
    "industrial carbon emission efficiency": "industrial carbon emission efficiency",
    "industrial emissions": "industrial emissions",
    "industrial energy-consumption": "industrial energy consumption",
    "industrial pollution": "industrial pollution",
    "industrial policy": "industrial policy",
    "industrial tree plantations": "industrial tree plantations",
    "energy governance": "energy governance",
    "energy policy network": "energy policy network",
    "energy politics": "energy politics",

    # tecnologías de información (se usan en gobernanza/monitoreo)
    "information and communication technologies": "information and communication technologies",

    # institucional / gobernanza
    "institutional adaptation": "institutional adaptation",
    "institutional complexity": "institutional complexity",
    "institutional fragmentation": "institutional fragmentation",
    "institutional governance": "institutional governance",
    "institutional logics": "institutional logics",
    "institutional quality and environmental regulation": "institutional quality and environmental regulation",

    # gestión integrada
    "integrated coastal management": "integrated coastal management",
    "integrated coastal zone management": "integrated coastal zone management",
    "integrated water resources management (iwrm)": "integrated water resources management",
    "integrated water resources management": "integrated water resources management",
    "integrative environmental governance": "integrative environmental governance",

    # internacional – derecho y regímenes
    "international biodiversity law": "international biodiversity law",
    "international climate policy": "international climate policy",
    "international climate regime": "international climate regime",
    "international commercial arbitration": "international commercial arbitration",
    "international court of justice (icj)": "international court of justice",
    "international courts": "international courts",
    "international customs law": "international customs law",
    "international development cooperation": "international development cooperation",
    "international diplomacy": "international diplomacy",
    "international economic law": "international economic law",
    "international environmental law history": "international environmental law history",
    "international environmental laws": "international environmental law",
    "international environmental law": "international environmental law",
    "international environmental policy": "international environmental policy",
    "international environmental regime": "international environmental regime",
    "international environmental regimes": "international environmental regimes",
    "international environmental regulation": "international environmental regulation",
    "international environmental treaty": "international environmental treaty",
    "international fisheries law": "international fisheries law",
    "international forest regime": "international forest regime",
    "international humanitarian law": "international humanitarian law",
    "international investment agreements": "international investment agreements",
    "international investment law": "international investment law",
    "international law (public)": "public international law",
    "international law and politics": "international law and politics",
    "international law commission": "international law commission",
    "international legal frameworks": "international legal framework",
    "international legal framework": "international legal framework",
    "international legal regime": "international legal regime",
    "international organization for standardization (iso)": "iso",
    "international political economy of climate change": "international political economy of climate change",
    "international principles": "international principles",
    "international refugee law": "international refugee law",
    "international regimes": "international regimes",
    "international relations and politics": "international relations and politics",
    "international river-basins": "international river basins",
    "international rivers": "international rivers",
    "international seabed authority (isa)": "international seabed authority",
    "international shipping": "international shipping",
    "international trade law": "international trade law",
    "international tribunal for the law of the sea": "itlos",
    "international tribunals": "international tribunals",
    "international watercourse": "international watercourse",

    # justicia climática / ambiental
    "intersectional climate justice": "intersectional climate justice",
    "just transition": "just transition",
    "justicia ambiental": "environmental justice",

    # ISO / estándares ambientales
    "iso-14001": "iso 14001",
    "iso 14001": "iso 14001",

    # conocimiento y política
    "knowledge-based policy": "knowledge-based policy",
    "knowledge governance": "knowledge governance",
    "knowledge production for sustainable development": "knowledge production for sustainable development",

    # tierra / gobernanza de recursos
    "land governance": "land governance",
    "land green use efficiency": "land green use efficiency",
    "land policy": "land policy",
    "land reform": "land reform",
    "land rights": "land rights",
    "land sustainability": "land sustainability",
    "land tenure": "land tenure",
    "land trusts": "land trusts",

    # derecho del mar / convenios
    "law of the sea convention": "united nations convention on the law of the sea",
    "minamata convention": "minamata convention",

    # gobernanza local / multinivel
    "local and regional governance": "local and regional governance",
    "local climate politics": "local climate politics",
    "local adaptation planning": "local adaptation planning",
    "local environmental governance": "local environmental governance",
    "local environmental strategies": "local environmental strategies",
    "local government annual report": "local government annual report",
    "local governments": "local governments",
    "local governments with limited resources": "local governments",
    "local public institutions": "local public institutions",
    "local sea-level rise adaptation": "local sea-level rise adaptation",

    # low-carbon (variante nuevas)
    "low-carbon cities": "low carbon cities",
    "low-carbon city initiative": "low carbon city initiative",
    "low-carbon city pilot": "low carbon city pilot",
    "low-carbon city pilots": "low carbon city pilot",
    "low-carbon economic transformation": "low carbon economic transformation",
    "low-carbon governance": "low carbon governance",
    "low-carbon industries": "low carbon industries",
    "low-carbon innovation": "low carbon innovation",
    "low-carbon patents": "low carbon patents",
    "low-carbon performance": "low carbon performance",
    "low-carbon policy": "low carbon policy",
    "low-carbon policy intensity": "low carbon policy intensity",
    "low-carbon technical efficiency": "low carbon technical efficiency",
    "low-carbon technological innovation": "low carbon technological innovation",
    "low-carbon technology": "low carbon technology",
    "low-carbon urban policy": "low carbon urban policy",
    "low -carbon pilot cities policy": "low carbon pilot cities policy",
    "low carbon": "low carbon",
    "low carbon emission": "low carbon emissions",
    "low emission": "low emissions",

    # derecho ambiental marino
    "marine environment law": "marine environmental law",
    "marine fishery policy": "marine fishery policy",

    # regulación ambiental
    "market-based environmental regulations": "market-based environmental regulation",
    "methods and models of environmental regulation": "environmental regulation models",

    # minería
    "mining law": "mining law",
    "mining politics": "mining politics"
}







def reemplazar_palabras_clave(column, diccionario_reemplazo):
    """
    Recorre cada celda de la columna, separa los términos (suponiendo que estén separados por ';'),
    y reemplaza aquellos que coincidan (ignorando mayúsculas/minúsculas) por el valor correspondiente del diccionario.
    """
    def process_cell(cell):
        # Si la celda es una cadena, separamos usando el delimitador ';'
        if isinstance(cell, str):
            terminos = [termino.strip() for termino in cell.split(';') if termino.strip()]
        # Si ya es una lista, la usamos directamente
        elif isinstance(cell, list):
            terminos = [str(termino).strip() for termino in cell if str(termino).strip()]
        else:
            terminos = []
        
        terminos_modificados = []
        for termino in terminos:
            # Convertimos el término a minúsculas para la comparación
            termino_lower = termino.lower()
            if termino_lower in diccionario_reemplazo:
                # Reemplazamos por el valor definido en el diccionario
                terminos_modificados.append(diccionario_reemplazo[termino_lower])
            else:
                terminos_modificados.append(termino)
        return '; '.join(terminos_modificados)
    
    return column.apply(process_cell)
print("Antes (recuento únicos):")
for c in KW_COLS:
    if c in df.columns:
        nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
        print(f"  {c}: {nuniq}")

# Aplicar la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = reemplazar_palabras_clave(df['Index Keywords'], palabras_clave_reemplazo)
df['Author Keywords'] = reemplazar_palabras_clave(df['Author Keywords'], palabras_clave_reemplazo)
#df['bothKeywords'] =  reemplazar_palabras_clave(df['bothKeywords'], palabras_clave_reemplazo)
# --- A PARTIR DE AQUÍ, EL CÓDIGO NUEVO PARA REEMPLAZOS PARCIALES ---
print("\nDespués (recuento únicos):")
for c in KW_COLS:
    if c in df.columns:
        nuniq = df[c].dropna().str.split(';').explode().str.strip().replace("", pd.NA).dropna().nunique()
        print(f"  {c}: {nuniq}")
def reemplazar_parciales(column, patrones):
    """
    Recorre cada celda de la columna, y por cada patrón (regex) en 'patrones',
    realiza la sustitución indicada.
    - 'patrones' debe ser una lista de tuplas (pattern, replacement).
    - Se ignoran mayúsculas/minúsculas (flags=re.IGNORECASE).
    """
    def process_cell(cell):
        if isinstance(cell, str):
            # Aplica todos los patrones de reemplazo parcial
            for patron, nuevo_texto in patrones:
                cell = re.sub(patron, nuevo_texto, cell, flags=re.IGNORECASE)
        return cell

    return column.apply(process_cell)

# Ejemplo de un arreglo de reemplazos parciales
# Cada tupla es (expresión_regular, texto_reemplazo)
# Aquí solo se incluye 'datum' -> 'data', pero puedes añadir más.
patrones_parciales = [
    (r'datum', 'data'),  # Reemplaza 'datum' donde aparezca (ignora mayúsculas)
    # Si necesitas más reemplazos parciales:
    # (r'algunaSubcadena', 'otroTexto'),
    # (r'pattern', 'replacement'),
    # ...
]
# 2) Después, los reemplazos parciales:
#df['bothKeywords'] = reemplazar_parciales(df['bothKeywords'], patrones_parciales)
df['Index Keywords'] = reemplazar_parciales(df['Index Keywords'], patrones_parciales)
df['Author Keywords'] = reemplazar_parciales(df['Author Keywords'], patrones_parciales)
# Guardar el DataFrame modificado en un nuevo archivo CSV
#df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)

df.to_csv(r"G:\Mi unidad\2025\Master  FRANCISCO MARCELO ALVARADO PORRAS\data\datawos_scopusreplace.csv", index=False)
print("Palabras clave reemplazadas y nuevo archivo guardado.")