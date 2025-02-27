# pip install wordcloud transformers spacy scikit-learn nltk gensim
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords



import re
import unicodedata
from gensim.models.phrases import Phrases, Phraser
import spacy

# Descargar stopwords y cargar modelo spaCy
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_lg")
# Descargar el paquete WordNet de NLTK si no está instalado
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Inicializa el lematizador
lemmatizer = WordNetLemmatizer()
# Lista de frases clave a preservar
key_phrases = [
    "job performance", "work performance", "employee performance",
    "task performance", "individual performance", "workplace performance",
    "human resource management", "organizational psychology",
    "employee well-being", "artificial intelligence",
    "international collaboration", "organizational behavior",
    "work engagement", "leadership effectiveness",
    "work productivity", "team performance",
    "employee satisfaction", "motivation at work",
    "psychological safety", "performance appraisal",
    "talent management", "workplace innovation",
    "collaborative networks", "bibliometric analysis",
    "emerging trends"
]

# Lista de palabras irrelevantes
irrelevant_words = [
   "rights", "reserved", "Emerald", "Publishing", "Limited", "author", 
    "found", "however", "also", "study", "data", "results", "analysis", 
    "questionnaire", "sample", "test", "authors", "findings", "method", 
    "design", "hypotheses", "companies", "COVID", "education", 
    "career", "addition", "future", "conducted", "may", "new", "show", 
    "revealed", "examined", "explore", "provides", "important", "aims", 
    "approach", "based", "framework", "introduction", "purpose", "context", 
    "process", "article", "paper", "used", "using", "one", "two", "three",
    "relationship", "research", "effect", "organization", "model", "management",
 "institution", "india", "us", "smes", "national", "nigeria", "province",
    "expatriate", "nurse", "bank", "sector", "male", "hotel", "enterprise",
    "hospital", "workplace_spirituality", "frontline_employees", "global",
    "period", "flexibility", "carried", "international", "finding",
    "data_collected", "rights_reserved", "study_aims", "results_showed",
    "present_study", "results_show", "social_media", "public_sector",
    "purpose_study", "covid_pandemic", "significant_effect",
    "research_limitationsimplications", "study_also", "results_study",
    "positively_related", "current_study", "regression_analysis",
    "positive_relationship", "mediating_effect", "positive_effect",
    "study_examines", "social_capital", "moderating_effect",
    "results_indicate", "originalityvalue_study", "future_research",
    "five", "jp", "determinant", "positive_impact", "data_collection",
    "social_support", "mindfulness", "item", "considering", "lower",
    "situation", "include", "higher_education", "intervention",
    "power", "suggested", "made", "job_performance_among",
    "positive_significant_effect", "hr", "student", "findings_suggest",
    "city", "particularly", "others", "possible", "address",
    "findings_study", "standard", "reduce", "essential", "eg",
    "studied", "emerald_group_publishing_limited", "position",
    "sale", "appropriate", "participation", "economic", "degree",
    "interest", "crucial", "stressor", "question", "innovative_work_behavior",
    "limitation", "shown", "received", "job_characteristics", "promotion",
    "po", "likely", "turnover_intention", "resilience", "decision",
    "limited_trading_taylor_francis", "social_exchange_theory",
    "psychological_wellbeing", "copyright", "region", "findings_results",
    "rate", "practitioner", "following", "often", "information_technology",
    "performance_assessment", "environmental", "lack", "strength",
    "workplace_ostracism", "predicting", "jordan", "worklife_balance",
    "authentic_leadership", "nature", "behavioral", "assessed",
    "score", "agency", "feeling", "still", "significant_relationship",
    "study", "system", "employer", "contextual_performance", "sample", 
    "turn", "element", "indicator", "taiwan", "path_analysis", 
    "conceptual", "following", "reliability", "school", "region", 
    "sampling", "economy", "identified", "colleague", "psychology", 
    "practical_implications", "statistical", "previous", "provided", 
    "individual_work_performance", "establish", "technical", 
    "previous_studies", "response", "job_satisfaction", 
    "criterion", "setting", "scope", "role_ambiguity", 
    "academic", "feedback", "exploration", "explain", "workplace",
    "providing", "overall", "exploring", "interview", "case", 
    "career", "study_conducted", "service_quality", "respectively", 
    "external", "reveal", "various", "established", "university",
    "implementation", "gap", "explored", "intervention", 
    "ethical_leadership", "organizational_learning", "objective", 
    "sustainable_development", "qualitative", "recruitment", 
    "findings", "behavioral", "utilizing"
,  
    
    "institution", "india", "us", "smes", "national", "nigeria", "province", 
   "expatriate", "nurse", "bank", "sector", "male", "hotel", "enterprise", 
   "hospital", "workplace_spirituality", "frontline_employees", "global", 
   "period", "flexibility", "carried", "international", "finding", 
   "data_collected", "rights_reserved", "study_aims", "results_showed", 
   "present_study", "results_show", "social_media", "public_sector", 
   "purpose_study", "covid_pandemic", "data_analysis", "significant_effect", 
   "research_limitationsimplications", "study_also", "results_study", 
   "positively_related", "current_study", "regression_analysis", 
   "positive_relationship", "mediating_effect", "positive_effect", 
   "study_examines", "social_capital", "moderating_effect", 
   "results_indicate", "originalityvalue_study", "future_research", 
   "five", "jp", "determinant", "positive_impact", "data_collection", 
   "social_support", "mindfulness", "item", "considering", "lower", 
   "situation", "include", "higher_education", "intervention", 
   "power", "suggested", "made", "job_performance_among", 
   "positive_significant_effect", "hr", "student", "findings_suggest", 
   "city", "particularly", "others", "possible", "address", 
   "findings_study", "standard", "reduce", "essential", "eg", 
   "studied", "emerald_group_publishing_limited", "position", 
   "sale", "appropriate", "participation", "economic", "degree", 
   "interest", "crucial", "stressor", "question", "innovative_work_behavior", 
   "limitation", "shown", "received", "job_characteristics", "promotion", 
   "po", "likely", "turnover_intention", "resilience", "decision", 
   "leadermember_exchange", "limited_trading_taylor_francis", 
   "social_exchange_theory", "psychological_wellbeing", "copyright", 
   "region", "findings_results", "structural_equation", "rate", 
   "practitioner", "following", "often", "information_technology", 
   "performance_assessment", "environmental", "lack", "strength", 
   "workplace_ostracism", "predicting", "jordan", "worklife_balance", 
   "authentic_leadership", "nature", "behavioral", "assessed", 
   "score", "agency", "feeling", "still", "significant_relationship", 
   "job_performance_job_satisfaction", "among_employees", 
   "personality_traits", "selection", "employees_working", 
   "expectation", "sustainability", "leadership_styles", 
   "training_development", "intrinsic_motivation", "prediction", 
   "component", "exploring", "human_capital", "work_stress", 
   "included", "focused", "intention", "since", "predict", 
   "job_security", "voice", "hr_practices", "enhanced", 
   "hrm_practices", "employer", "planning", "particular", 
   "employee_wellbeing", "individual_work_performance", 
   "inrole_performance", "achievement", "work_discipline", "create", 
   "organizational_identification", "justice", "complex", 
   "employee_satisfaction", "background", "major", "community", 
   "job_satisfaction_job_performance", "unit", "vietnam", "play", 
   "basis", "sample_employees", "coworkers", "positively_associated", 
   "anxiety", "appraisal", "workfamily_conflict", "without", 
   "job_demands", "view", "might", "connection", "unique", "highly", 
   "follower", "telework", "initiative", "consider", "hence", 
   "action", "influencing", "performance_evaluation", "discipline", 
   "become", "like", "concern", "psychological_safety", "purpose_paper", 
   "significant_impact", "study_examined", "employee_creativity", 
   "target", "demonstrate", "recommendation", "day", "despite", 
   "responsibility", "promote", "study_conducted", "workplace_performance", 
   "et_al", "focusing", "ethical", "belief", "subject", "much", 
   "performed", "metaanalysis", "yet", "mediates_relationship", 
   "established", "v", "creative", "investigating", "indirect_effect", 
   "reported", "study_aimed", "equation_modeling", "data_analyzed", 
   "structural_equation_modeling_sem", "motivational", "multilevel", 
   "results_indicated", "usage", "element", "study_contributes", 
   "path_analysis", "hotel_employees", "hospitality_industry", 
   "status", "multiple_regression", "colleague", "self", 
   "employee_motivation", "study_investigates", "results_revealed", 
   "function", "respectively", "experiment", "software", "necessary", 
   "empowering_leadership", "analyzed_using", "gap", "foster", 
   "explored", "moderation", "discussion", "structural", "direction", 
   "affective", "proactive_personality", "practical", "variance", 
   "best", "elsevier_ltd", "experienced", "taiwan",
    "rights_reserved", "data_collected", "designmethodologyapproach", 
    "study_aims", "results_showed", "present_study", "results_show", 
    "social_media", "public_sector", "purpose_study", 
    "job_performance_job_satisfaction", "covid_pandemic", 
    "significant_effect", "knowledge_management", "taylor_francis", 
    "test_hypotheses", "study_provides", "political_skill", 
    "mental_health", "banking_sector", "working_conditions", 
    "positive_influence", "average", "hand", "along", "sem", 
    "user", "survey_data", "rather", "informa_uk", 
    "observed", "simultaneously", "service_quality", "case_study", 
    "driver", "superior", "presented", "previous_studies", 
    "external", "administrative", "engaged", "sample_size", 
    "el", "oc", "confirmed", "enhances", "plan", 
    "recommended", "accuracy", "today", "primary_data", 
    "novel", "salary", "science", "conduct", "concluded", 
    "occupational_stress", "point", "administration", 
    "sampling", "rated", "reason", "recent", 
    "research_conducted", "distribution", "bias", 
    "corporate_social_responsibility", "educational", 
    "technological", "phenomenon", "competition", 
    "adjustment", "population_study", "negative_relationship", 
    "facet", "pattern", "seek", "contrast", "cause", 
    "cognition", "upon", "weak", "implemented", 
    "loyalty", "chain", "status", "alternative", 
    "know", "drawn", "frontline", "fully",    "result", "literature", "value", "time", "industry", "implication", "change",
    "survey", "dimension", "different", "group", "stress", "perspective",
    "firm", "wellbeing", "learning", "culture", "environment", "working",
    "related", "experience", "higher", "staff", "business", "furthermore",
    "communication", "relation", "ie", "importance", "association", "organisation",
    "scale", "emerald_publishing_limited", "understanding", "ocb", "indonesia",
    "thus", "mechanism", "teacher", "information", "evaluation", "condition",
    "characteristic", "technology", "china", "health", "control", "hypothesis",
    "first", "construct", "n", "type", "evidence", "find", "link", "policy",
    "moreover", "therefore", "personal", "potential", "professional",
    "provide", "better", "negative", "could", "age", "enhancing", "interaction",
    "low", "difference", "attitude", "including", "correlation", "public",
    "tool", "considered", "via", "especially", "general", "term",
    "orientation", "good", "regarding", "issue", "employed", "moderator",
    "moderated", "managerial", "csr", "less", "area", "proposed",
    "field", "key", "several", "due", "second", "mean", "finally",
    "although", "internal", "specific", "many", "digital", "success",
    "gender", "office", "conclusion", "government", "lead", "demand",
    "network", "population", "make", "de", "present", "concept", "greater",
    "number", "total", "namely", "selected", "suggest", "identify",
    "state", "style", "affected", "expected", "whereas", "pay",
    "antecedent", "validity", "emotion", "instrument", "form", "toward",
    "problem", "cultural", "attention", "trait", "critical", "country",
    "compared", "set", "library", "consequence", "family", "drawing",
    "theoretical", "discussed", "extent", "growth", "must", "developing",
    "adopted", "year", "assess", "methodology", "industrial", "according",
    "testing", "online", "pandemic", "review", "crisis", "life",
    "source", "relative", "suggests", "achieve", "empirical", "measured",
    "member", "diversity", "aim", "corporate", "part", "financial",
    "multiple", "building", "directly", "offer", "regression", "workrelated",
    "interpersonal", "project", "quantitative","variable", "help","woker"
]

# Preservar frases clave reemplazando espacios por guiones bajos
def preserve_phrases(text, key_phrases):
    for phrase in key_phrases:
        text = text.replace(phrase, phrase.replace(" ", "_"))
    return text

# Limpiar y preprocesar texto
def preprocess_text(text):
    text = preserve_phrases(text, key_phrases)
    text = re.sub(r"[^a-zA-Z\s_]", "", text)  # Elimina caracteres especiales
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # Remueve acentos
    words = [word for word in text.split() if word.lower() not in stop_words and word.lower()]
    return " ".join(words)

# Cargar datos
consolidado = "G:\\Mi unidad\\2024\\Master Solis Granda Luis Eduardo\\data\\wos_scopus.csv"
dataframe = pd.read_csv(consolidado)

# Manejar valores nulos
columns_to_check = ['Abstract', 'Title', 'Author Keywords', 'Index Keywords']
for col in columns_to_check:
    dataframe[col] = dataframe[col].fillna("").astype(str)

# Combinar y preprocesar textos
text_data = (dataframe['Abstract'] + " " + dataframe['Title'] + " " +
             dataframe['Author Keywords'] + " " + dataframe['Index Keywords']).apply(preprocess_text)

# Tokenizar textos para bigrams y trigrams
tokenized_texts = [doc.split() for doc in text_data]
bigram_model = Phrases(tokenized_texts, min_count=3, threshold=5)
bigram_phraser = Phraser(bigram_model)
trigram_model = Phrases(bigram_phraser[tokenized_texts], min_count=5, threshold=10)
trigram_phraser = Phraser(trigram_model)

# Aplicar bigrams y trigrams
bigram_texts = [bigram_phraser[doc] for doc in tokenized_texts]
trigram_texts = [trigram_phraser[doc] for doc in bigram_texts]

# Combinar textos con trigrams para conteo final
final_texts = [" ".join(doc) for doc in trigram_texts]
filtered_text = " ".join(final_texts)
word_counts = Counter(filtered_text.split())

# Filtrar palabras con frecuencia mayor a 5
filtered_word_counts = {word: freq for word, freq in word_counts.items() if freq > 5}




# Aplicar lematización y filtrar palabras irrelevantes después de la lematización
lemmatized_word_counts = {}
for word, freq in filtered_word_counts.items():
    # Lematizar palabras individuales
    if " " not in word:  # Solo lematizar palabras individuales
        lemmatized_word = lemmatizer.lemmatize(word.lower())
    else:
        lemmatized_word = word  # Dejar frases clave intactas
    
    # Filtrar palabras irrelevantes después de la lematización
    if lemmatized_word not in irrelevant_words:
        lemmatized_word_counts[lemmatized_word] = lemmatized_word_counts.get(lemmatized_word, 0) + freq
# Mostrar las 20 palabras más frecuentes
print("Top 20 palabras clave con frecuencia > 5:")
for word, freq in Counter(lemmatized_word_counts).most_common(300):
    print(f"{word}: {freq} veces")

# Generar nube de palabras
print("Generando nube de palabras...")
wordcloud = WordCloud(
    stopwords='english',
    background_color='white',
    max_words=300,
    width=1960,
    height=1080,
    colormap="viridis"
).generate_from_frequencies(lemmatized_word_counts)

plt.figure(figsize=(15, 8), dpi=120)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Global Word Cloud with Key Phrases', fontsize=20, color='darkblue', pad=20)
plt.tight_layout()
plt.show()
