# -*- coding: utf-8 -*-
"""
02_eda_topics_embeddings_plus_v4.py

Novedades vs v3:
- Fix MatplotlibDeprecationWarning (get_cmap -> matplotlib.colormaps.get_cmap).
- Fix KeyError al etiquetar nodos de networkx (filtra etiquetas por nodos presentes).
- Nueva: Temporal Semantic Shift Map (flechas entre periodos A y B por keyword).
- Mantiene: LDA, embeddings, UMAP/PCA+KMeans con etiquetas mejoradas, redes y Thematic Map,
  dendrogramas, treemap, timeline, barras técnicas/aplicaciones, similitud coseno.

Requisitos core: numpy, pandas, matplotlib, scikit-learn, wordcloud.
Opcionales (auto-try): seaborn, sentence-transformers, umap-learn, scipy, squarify, pyLDAvis, networkx.
"""

import os, ast, json, math, itertools, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# opcional: seaborn
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans

# opcionales
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import squarify  # treemap
    HAS_SQUARIFY = True
except Exception:
    HAS_SQUARIFY = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import pyLDAvis, pyLDAvis.sklearn
    PLDV_OK = True
except Exception:
    PLDV_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except Exception:
    ST_OK = False

# grafos
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False


# ================== CONFIG ==================
INPUT  = r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1_cleanfinalfinal.csv"
OUTDIR = str(Path(INPUT).with_name("eda_topics_embeddings_outputs"))
os.makedirs(OUTDIR, exist_ok=True)

COL_TITLE = "Title"
COL_TEXT  = "text_clean"         # texto limpio para PLN
COL_YEAR  = "year"
COL_KWU   = "Keywords Unified"   # texto con separador ';'
COL_KWL   = "keywords_list"      # lista (string); si falta, la derivamos de KWU
SEED      = 42

# Parámetros principales
N_TOP_FREQ    = 25
MIN_DF_TEXT   = 5
MIN_DF_KW     = 1
N_TOPICS_LDA  = 6
KMEANS_K      = 6
N_NEIGHBORS   = 15
MIN_DIST      = 0.1
TOPK_SIM      = 5

# Grafo de co-ocurrencia
TOP_KW_FOR_GRAPH   = 80     # top keywords por frecuencia a incluir en grafo
MIN_EDGE_WEIGHT    = 2      # umbral mínimo de co-ocurrencia para dibujar aristas

# Thematic map
THEME_LABEL_TOPN   = 3

# Temporal Semantic Shift Map
# Define dos ventanas de años (ejemplo parecido a 2011 vs 2015 que mostraste)
PERIOD_A = (2011, 2013)   # inclusive
PERIOD_B = (2014, 2016)   # inclusive
SHIFT_TOP_KW = 60         # cuántas keywords (más frecuentes) evaluar
SHIFT_MIN_DOCS_PER_PERIOD = 3  # requerimiento mínimo de docs por periodo para esa keyword

# Diccionarios regex
FAMILIES = {
    "Transformer": r"\btransformer(s)?\b|attention|self[-\s]?attention|encoder[-\s]?decoder",
    "BERT": r"\bbert\b|roberta|albert|distilbert",
    "GPT": r"\bgpt[-\s]?\d*|chatgpt\b",
    "RNN/LSTM": r"\brnn\b|lstm|gru",
    "Attention (genérico)": r"\battention\b|self[-\s]?attention|multi[-\s]?head"
}
TECHNIQUES = {
    "Tokenization": r"\btokeni[sz]ation|wordpiece|bpe\b",
    "Embeddings": r"\bembedding(s)?\b|word2vec|glove|fasttext",
    "Fine-Tuning": r"\bfine[-\s]?tuning|instruction[-\s]?tuning|sft\b",
    "Transfer Learning": r"\btransfer learning\b|pre[-\s]?train(ed|ing)",
    "RAG": r"\brag\b|retrieval[-\s]?augmented",
    "Vector Search": r"\bvector (db|database|store|search)|faiss\b"
}
APPLICATIONS = {
    "Text Classification": r"\b(classification|classify|classifier)\b",
    "Sentiment Analysis": r"\bsentiment\b",
    "Question Answering": r"\b(question answering|qa)\b",
    "Translation": r"\btranslation|translate|machine translation\b",
    "Summarization": r"\bsummarization|summari[sz]e\b",
    "Clustering/Topic": r"\bclustering|topic modeling|lda\b",
    "Information Retrieval": r"\binformation retrieval|semantic search\b"
}

# ========== utilidades ==========
def savefig(path, dpi=200, tight=True):
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def parse_kw_list(x):
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return [str(k).strip() for k in v if str(k).strip()]
    except Exception:
        pass
    return []

def ensure_keywords_list(df):
    """Si no existe keywords_list, derivarlo desde 'Keywords Unified' (separador ';')."""
    if COL_KWL not in df.columns:
        df[COL_KWL] = df.get(COL_KWU, "").fillna("").apply(
            lambda s: str([t.strip() for t in str(s).split(";") if t.strip()])
        )
    return df

def bool_col_from_regex(series, rx):
    patt = re.compile(rx, flags=re.IGNORECASE)
    return series.fillna("").astype(str).apply(lambda s: 1 if patt.search(s) else 0)

# ================== 1) CARGA ==================
df = pd.read_csv(INPUT, dtype=str, encoding="utf-8")
df = ensure_keywords_list(df)

if COL_TEXT not in df.columns:
    raise ValueError(f"Columna {COL_TEXT} no encontrada en el CSV.")

df[COL_TEXT] = df[COL_TEXT].fillna("").astype(str)
df["kw_list_parsed"] = df[COL_KWL].apply(parse_kw_list)
if COL_YEAR in df.columns:
    df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
print(f"Registros cargados: {len(df)}")

# Limpieza rápida: quita frases vacías tipo "no abstract available"
df[COL_TEXT] = df[COL_TEXT].str.replace(r"\bno\s+abstract\s+available\b", "", regex=True, flags=re.I)
min_tokens = 10
df = df[df[COL_TEXT].str.split().str.len().fillna(0) >= min_tokens].reset_index(drop=True)

# ================== 2) EXPLORACIÓN BÁSICA ==================
# WordCloud (texto)
all_text = " ".join(df[COL_TEXT].tolist()).strip()
if all_text:
    wc = WordCloud(width=1600, height=800, background_color="white").generate(all_text)
    plt.figure(figsize=(14,7)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title("WordCloud – Title+Abstract (text_clean)")
    savefig(Path(OUTDIR, "wordcloud_title_abstract_text.png"))

# WordCloud (keywords)
all_kws = []
for kws in df["kw_list_parsed"]:
    all_kws.extend([k.lower() for k in kws if k])
kw_text = " ".join(all_kws).strip()
if kw_text:
    wc2 = WordCloud(width=1600, height=800, background_color="white").generate(kw_text)
    plt.figure(figsize=(14,7)); plt.imshow(wc2, interpolation="bilinear"); plt.axis("off")
    plt.title("WordCloud – Keywords (keywords_list)")
    savefig(Path(OUTDIR, "wordcloud_keywords.png"))

# Frecuencias – Texto
vec_text = CountVectorizer(max_df=0.8, min_df=MIN_DF_TEXT, ngram_range=(1,2))
X_text = vec_text.fit_transform(df[COL_TEXT])
terms_text = vec_text.get_feature_names_out()
freqs_text = np.asarray(X_text.sum(axis=0)).ravel()
freq_text_df = pd.DataFrame({"term": terms_text, "count": freqs_text}).sort_values("count", ascending=False)
freq_text_df.to_csv(Path(OUTDIR, "term_frequencies_text.csv"), index=False, encoding="utf-8")
top_text = freq_text_df.head(N_TOP_FREQ)
plt.figure(figsize=(12,6)); plt.barh(top_text["term"][::-1], top_text["count"][::-1])
plt.title("Top términos (Title+Abstract)")
savefig(Path(OUTDIR, "top_terms_text.png"))

# Frecuencias – Keywords
vec_kw = CountVectorizer(tokenizer=lambda s: s.split("|||"), lowercase=True, min_df=MIN_DF_KW)
kw_joined = ["|||".join([k.lower() for k in ks]) if ks else "" for ks in df["kw_list_parsed"]]
X_kw = vec_kw.fit_transform(kw_joined)
terms_kw = vec_kw.get_feature_names_out()
freqs_kw = np.asarray(X_kw.sum(axis=0)).ravel()
freq_kw_df = pd.DataFrame({"keyword": terms_kw, "count": freqs_kw}).sort_values("count", ascending=False)
freq_kw_df.to_csv(Path(OUTDIR, "keyword_frequencies.csv"), index=False, encoding="utf-8")
top_kw = freq_kw_df.head(N_TOP_FREQ)
plt.figure(figsize=(12,6)); plt.barh(top_kw["keyword"][::-1], top_kw["count"][::-1])
plt.title("Top Keywords (Index+Author)")
savefig(Path(OUTDIR, "top_keywords.png"))

# ================== 3) LDA ==================
lda = LatentDirichletAllocation(n_components=N_TOPICS_LDA, random_state=SEED, learning_method="batch")
lda.fit(X_text)
def top_terms_per_topic(model, feature_names, topn=10):
    out = []
    for idx, comp in enumerate(model.components_):
        top_idx = comp.argsort()[-topn:][::-1]
        out.append([feature_names[i] for i in top_idx])
    return out
lda_topics = top_terms_per_topic(lda, terms_text, topn=10)
pd.DataFrame([{"topic": i+1, "top_terms": ", ".join(t)} for i, t in enumerate(lda_topics)]) \
  .to_csv(Path(OUTDIR, "lda_topics_top_terms.csv"), index=False, encoding="utf-8")
if PLDV_OK:
    vis = pyLDAvis.sklearn.prepare(lda, X_text, vec_text, mds='tsne')
    pyLDAvis.save_html(vis, str(Path(OUTDIR, "lda_pyldavis.html")))

# ================== 4) EMBEDDINGS ==================
if ST_OK:
    model_name = "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(model_name)
    docs = df[COL_TEXT].fillna("").tolist()
    doc_vecs = st_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    doc_vecs = np.asarray(doc_vecs)
else:
    tfidf_fallback = TfidfVectorizer(max_df=0.8, min_df=MIN_DF_TEXT, ngram_range=(1,2))
    doc_vecs = tfidf_fallback.fit_transform(df[COL_TEXT])
    if hasattr(doc_vecs, "toarray"):
        doc_vecs = doc_vecs.toarray()
np.save(Path(OUTDIR, "doc_vectors.npy"), doc_vecs)

# ================== 5) DIAGNÓSTICO K + UMAP/PCA + KMEANS ==================
ks, inertias, sils = [], [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=SEED, n_init="auto").fit(doc_vecs)
    ks.append(k); inertias.append(km.inertia_); sils.append(silhouette_score(doc_vecs, km.labels_))
pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": sils}) \
  .to_csv(Path(OUTDIR, "k_diagnostics.csv"), index=False)

if 'umap' in globals() and HAS_UMAP:
    um = umap.UMAP(n_components=2, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, random_state=SEED)
    doc_2d = um.fit_transform(doc_vecs)
else:
    doc_2d = PCA(n_components=2, random_state=SEED).fit_transform(doc_vecs)

kmeans = KMeans(n_clusters=KMEANS_K, random_state=SEED, n_init="auto")
labels = kmeans.fit_predict(doc_vecs)
df["cluster_kmeans"] = labels

# ================== Etiquetas por cluster (mejoradas) ==================
from sklearn.feature_extraction.text import TfidfVectorizer
BLOCK_TERMS = {
    "no","abstract","available","study","paper","result","results","approach","based","method","methods",
    "analysis","using","use","data","system","model","models","effect","effects","research","process",
    "application","applications","finding","findings","however","therefore","may","provide","propose",
    "online","technology","digital","leader","leadership","virtual","organization","organizational"
}
PRIORITY_TERMS = {
    "e-leadership","digital leadership","virtual leadership","higher education",
    "teacher","teachers","faculty",
    "well-being","wellbeing","burnout","stress",
    "performance","job performance","teaching performance",
    "workload","work engagement","job satisfaction",
    "student outcomes","learning outcomes",
    "technology-enhanced learning","distance education","online learning",
    "covid-19","post-pandemic",
    "leadership style","transformational leadership","leader–member exchange","lmx"
}
tfidf_vec = TfidfVectorizer(max_df=0.85, min_df=3, ngram_range=(1,2))
X_tfidf = tfidf_vec.fit_transform(df[COL_TEXT])
terms = np.array(tfidf_vec.get_feature_names_out())

# Top keywords por cluster (usamos X_kw)
Xkw_df = pd.DataFrame(X_kw.toarray(), columns=terms_kw)

cluster_text_rows, cluster_kw_rows = [], []
# preparar penalizaciones/bonos
term_index = {t:i for i,t in enumerate(terms)}
penalty = np.ones(len(terms)); bonus = np.ones(len(terms))
for t in BLOCK_TERMS:
    if t in term_index: penalty[term_index[t]] = 0.1
for t in PRIORITY_TERMS:
    if t in term_index: bonus[term_index[t]] = 1.4

for c in sorted(np.unique(labels)):
    idx = np.where(labels == c)[0]
    mean_tfidf = np.asarray(X_tfidf[idx].mean(axis=0)).ravel()
    score = mean_tfidf * penalty * bonus
    order = np.argsort(-score)

    top_terms = []
    for i in order:
        tt = terms[i]
        if tt in BLOCK_TERMS: continue
        if len(tt) < 2 or tt.isdigit(): continue
        top_terms.append(tt)
        if len(top_terms) >= 10: break

    sub_kw = Xkw_df.iloc[idx]
    top_kw = sub_kw.mean(axis=0).sort_values(ascending=False).head(10).index.tolist()

    cluster_text_rows.append({"cluster": int(c), "top_terms_text": ", ".join(top_terms[:10])})
    cluster_kw_rows.append({"cluster": int(c), "top_keywords": ", ".join(top_kw[:10])})

pd.DataFrame(cluster_text_rows).to_csv(Path(OUTDIR, "cluster_top_terms_text.csv"), index=False)
pd.DataFrame(cluster_kw_rows).to_csv(Path(OUTDIR, "cluster_top_keywords.csv"), index=False)

text_map = {d["cluster"]: d["top_terms_text"] for d in cluster_text_rows}
kw_map   = {d["cluster"]: d["top_keywords"]  for d in cluster_kw_rows}
cluster_label_map = {}
for c in sorted(np.unique(labels)):
    tshort = ", ".join(text_map[c].split(", ")[:3]) if c in text_map else ""
    kshort = ", ".join(kw_map[c].split(", ")[:2])   if c in kw_map else ""
    label = (tshort + (" | " + kshort if kshort else "")).strip(" |")
    cluster_label_map[c] = label if label else f"Cluster {c}"

plt.figure(figsize=(10,8))
palette = (plt.cm.tab10.colors if not HAS_SNS else sns.color_palette("tab10", n_colors=len(np.unique(labels))))
for c in sorted(np.unique(labels)):
    idx = labels == c
    plt.scatter(doc_2d[idx,0], doc_2d[idx,1], s=12, color=palette[c % len(palette)], label=f"C{c}")
for c in sorted(np.unique(labels)):
    idx = labels == c
    cx, cy = doc_2d[idx,0].mean(), doc_2d[idx,1].mean()
    plt.text(cx, cy, cluster_label_map[c], fontsize=9, weight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.7))
plt.legend(title="Clusters", loc="best")
plt.title("UMAP/PCA + KMeans — Etiquetas por cluster (TF-IDF + Keywords, filtradas)")
savefig(Path(OUTDIR, "umap_kmeans_labeled.png"))

df["cluster_label"] = [cluster_label_map[c] for c in df["cluster_kmeans"]]
df.to_csv(Path(OUTDIR, "dataset_with_clusters.csv"), index=False, encoding="utf-8")

# ================== 6) CO-OCURRENCIA DE KEYWORDS (CSV + GRAFO) ==================
edges_c, nodes_c = Counter(), Counter()
for kws in df["kw_list_parsed"]:
    kws_norm = [k.strip().lower() for k in kws if k and isinstance(k, str)]
    for k in kws_norm: nodes_c[k] += 1
    for a, b in itertools.combinations(sorted(set(kws_norm)), 2):
        edges_c[(a, b)] += 1

nodes_df = pd.DataFrame([{"id": k, "label": k, "weight": v} for k, v in nodes_c.items()])
edges_df = pd.DataFrame([{"source": s, "target": t, "weight": w} for (s, t), w in edges_c.items()])
nodes_df.to_csv(Path(OUTDIR, "gephi_nodes.csv"), index=False, encoding="utf-8")
edges_df.to_csv(Path(OUTDIR, "gephi_edges.csv"), index=False, encoding="utf-8")

if HAS_NX and len(nodes_df) > 0:
    # limitar a top keywords
    top_kw_ids = set(nodes_df.sort_values("weight", ascending=False).head(TOP_KW_FOR_GRAPH)["id"])
    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        if r["id"] in top_kw_ids:
            G.add_node(r["id"], weight=int(r["weight"]))
    for _, r in edges_df.iterrows():
        a, b, w = r["source"], r["target"], int(r["weight"])
        if w >= MIN_EDGE_WEIGHT and a in top_kw_ids and b in top_kw_ids:
            G.add_edge(a, b, weight=w)

    # comunidades
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        comm_map = {n:i for i,cset in enumerate(comms) for n in cset}
    except Exception:
        from networkx.algorithms.community import asyn_lpa_communities
        comms = list(asyn_lpa_communities(G))
        comm_map = {n:i for i,cset in enumerate(comms) for n in cset}

    pos = nx.spring_layout(G, seed=SEED, k=0.3, iterations=100)
    plt.figure(figsize=(12,9))
    cmap = matplotlib.colormaps.get_cmap("tab20")  # FIX deprecación
    node_sizes = [max(50, 20*G.nodes[n].get("weight",1)) for n in G.nodes()]
    node_colors = [cmap(comm_map.get(n,0)%20) for n in G.nodes()]
    edge_widths = [max(0.5, 0.8*G.edges[e]["weight"]) for e in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.25)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    # etiquetas: SOLO nodes “grandes” que están en pos
    weights_list = [G.nodes[x].get("weight",0) for x in G.nodes()]
    thr = np.percentile(weights_list, 75) if len(weights_list) else 0
    big_nodes = [n for n in G.nodes() if G.nodes[n].get("weight",0) >= thr and n in pos]
    nx.draw_networkx_labels(G, {n:pos[n] for n in big_nodes}, font_size=9)
    plt.axis("off")
    plt.title("Collaboration network of keywords (co-occurrence)")
    savefig(Path(OUTDIR, "keywords_collaboration_network.png"))

# ================== 7) THE MATIC MAP (communities) ==================
if HAS_NX and len(nodes_df) > 0:
    if 'G' not in locals():
        top_kw_ids = set(nodes_df.sort_values("weight", ascending=False).head(TOP_KW_FOR_GRAPH)["id"])
        G = nx.Graph()
        for _, r in nodes_df.iterrows():
            if r["id"] in top_kw_ids:
                G.add_node(r["id"], weight=int(r["weight"]))
        for _, r in edges_df.iterrows():
            a, b, w = r["source"], r["target"], int(r["weight"])
            if w >= MIN_EDGE_WEIGHT and a in top_kw_ids and b in top_kw_ids:
                G.add_edge(a, b, weight=w)

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
    except Exception:
        from networkx.algorithms.community import asyn_lpa_communities
        comms = list(asyn_lpa_communities(G))

    def internal_weight(sub_nodes):
        return sum(d.get("weight",1.0) for _,_,d in G.subgraph(sub_nodes).edges(data=True))

    def external_weight(sub_nodes):
        sset = set(sub_nodes); w = 0.0
        for u in sub_nodes:
            for v, d in G[u].items():
                if v not in sset:
                    w += d.get("weight",1.0)
        return w

    rows_tm = []
    for i, cset in enumerate(comms):
        cset = list(cset); n = len(cset)
        if n < 2:
            density = 0.0
        else:
            w_in = internal_weight(cset); possible = n*(n-1)/2
            density = 100.0 * (w_in / possible)
        cent = 10.0 * external_weight(cset)
        size = int(sum(nodes_c.get(k, 0) for k in cset))
        top_label_kw = sorted(cset, key=lambda k: nodes_c.get(k,0), reverse=True)[:THEME_LABEL_TOPN]
        label = ", ".join(top_label_kw)
        rows_tm.append({"theme_id": i, "label": label, "density": density, "centrality": cent, "size": size})

    tm_df = pd.DataFrame(rows_tm)
    tm_df.to_csv(Path(OUTDIR, "thematic_map_stats.csv"), index=False, encoding="utf-8")

    cx = tm_df["centrality"].median() if not tm_df["centrality"].empty else 0
    cy = tm_df["density"].median() if not tm_df["density"].empty else 0

    plt.figure(figsize=(12,8))
    plt.axvline(cx, color="gray", linestyle="--", alpha=0.6)
    plt.axhline(cy, color="gray", linestyle="--", alpha=0.6)
    sizes = (tm_df["size"].fillna(1).astype(float) ** 0.8) * 10.0
    plt.scatter(tm_df["centrality"], tm_df["density"], s=sizes, alpha=0.7)
    for _, r in tm_df.iterrows():
        plt.text(r["centrality"], r["density"], r["label"], fontsize=9, ha="center", va="center")
    xmin, xmax = plt.xlim(); ymin, ymax = plt.ylim()
    plt.text(xmin+(cx-xmin)*0.05, cy+(ymax-cy)*0.8, "Niche Themes", fontsize=10, color="gray")
    plt.text(cx+(xmax-cx)*0.6,   cy+(ymax-cy)*0.8, "Motor Themes", fontsize=10, color="gray")
    plt.text(xmin+(cx-xmin)*0.05,ymin+(cy-ymin)*0.1,"Emerging or\nDeclining Themes", fontsize=10, color="gray")
    plt.text(cx+(xmax-cx)*0.6,   ymin+(cy-ymin)*0.1,"Basic Themes", fontsize=10, color="gray")
    plt.xlabel("Relevance degree (Centrality)")
    plt.ylabel("Development degree (Density)")
    plt.title("Thematic Map (keywords communities)")
    savefig(Path(OUTDIR, "thematic_map.png"))

# ================== 8) TIMELINE POR FAMILIAS ==================
def multi_hot_from_dict(series_text, series_kw, rx_dict):
    out = {}
    for name, rx in rx_dict.items():
        m = bool_col_from_regex(series_text, rx) | bool_col_from_regex(series_kw, rx)
        out[name] = m
    return pd.DataFrame(out)

if COL_YEAR in df.columns and df[COL_YEAR].notna().any():
    fam_counts = []
    for fam, rx in FAMILIES.items():
        mask = bool_col_from_regex(df[COL_TEXT], rx) | bool_col_from_regex(df.get(COL_KWU, pd.Series([""]*len(df))), rx)
        tmp = df.loc[mask.astype(bool), [COL_YEAR]].copy()
        tmp["family"] = fam
        fam_counts.append(tmp)
    if len(fam_counts):
        fam_df = pd.concat(fam_counts, ignore_index=True)
        timeline = fam_df.dropna().groupby([COL_YEAR, "family"]).size().reset_index(name="count")
        timeline.to_csv(Path(OUTDIR, "timeline_families.csv"), index=False, encoding="utf-8")

        plt.figure(figsize=(12,6))
        for fam in timeline["family"].unique():
            sub = timeline[timeline["family"] == fam].sort_values(COL_YEAR)
            plt.plot(sub[COL_YEAR], sub["count"], marker="o", label=fam)
        plt.title("Evolución temporal por familia de modelos")
        plt.xlabel("Año"); plt.ylabel("# Documentos"); plt.legend()
        savefig(Path(OUTDIR, "timeline_families.png"))

mh_tech = multi_hot_from_dict(df[COL_TEXT], df.get(COL_KWU, pd.Series([""]*len(df))), TECHNIQUES)
mh_app  = multi_hot_from_dict(df[COL_TEXT], df.get(COL_KWU, pd.Series([""]*len(df))), APPLICATIONS)
mh_tech.to_csv(Path(OUTDIR, "multi_hot_techniques.csv"), index=False, encoding="utf-8")
mh_app.to_csv(Path(OUTDIR, "multi_hot_applications.csv"), index=False, encoding="utf-8")

tech_counts = mh_tech.sum().sort_values(ascending=False)
app_counts  = mh_app.sum().sort_values(ascending=False)
plt.figure(figsize=(10,5)); plt.barh(tech_counts.index[::-1], tech_counts.values[::-1])
plt.title("Técnicas detectadas (regex)")
savefig(Path(OUTDIR, "techniques_bar.png"))
plt.figure(figsize=(10,5)); plt.barh(app_counts.index[::-1], app_counts.values[::-1])
plt.title("Aplicaciones detectadas (regex)")
savefig(Path(OUTDIR, "applications_bar.png"))

# ================== 9) DENDROGRAMAS ==================
if HAS_SCIPY:
    # documentos (cosine)
    Z = linkage(pdist(doc_vecs, metric="cosine"), method="average")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, no_labels=True, count_sort=True)
    plt.title("Dendrograma de documentos (cosine, average linkage)")
    savefig(Path(OUTDIR, "dendrogram_documents.png"))

    # keywords (1 - jaccard)
    if len(terms_kw) > 1:
        Xkw_bin = (X_kw > 0).astype(int).toarray()
        Zkw = linkage(pdist(Xkw_bin.T, metric="jaccard"), method="average")
        plt.figure(figsize=(12, 5))
        dendrogram(Zkw, labels=terms_kw, leaf_rotation=90)
        plt.title("Dendrograma de keywords (1 - Jaccard)")
        savefig(Path(OUTDIR, "dendrogram_keywords.png"))

# ================== 10) TREEMAP DE KEYWORDS ==================
if HAS_SQUARIFY:
    topk_kw = freq_kw_df.head(50)
    sizes = topk_kw["count"].values
    labels_tm = [f"{k}\n{c}" for k, c in zip(topk_kw["keyword"], topk_kw["count"])]
    plt.figure(figsize=(12,7))
    squarify.plot(sizes=sizes, label=labels_tm, alpha=0.8)
    plt.axis("off"); plt.title("Treemap — Top Keywords")
    savefig(Path(OUTDIR, "treemap_keywords.png"))

# ================== 11) SIMILITUD COSENO (top vecinos) ==================
dists = pairwise_distances(doc_vecs, metric="cosine")
sims  = 1.0 - dists
rows = []
for i in range(len(df)):
    idx = np.argsort(-sims[i])
    topn = [j for j in idx if j != i][:TOPK_SIM]
    for j in topn:
        rows.append({
            "doc_i": i, "doc_j": j,
            "sim_cosine": float(sims[i, j]),
            "title_i": (df.get(COL_TITLE, pd.Series([""])).iloc[i] if COL_TITLE in df.columns else ""),
            "title_j": (df.get(COL_TITLE, pd.Series([""])).iloc[j] if COL_TITLE in df.columns else "")
        })
pd.DataFrame(rows).to_csv(Path(OUTDIR, "doc_similarity_topK.csv"), index=False, encoding="utf-8")

# ================== 12) TEMPORAL SEMANTIC SHIFT MAP ==================
if COL_YEAR in df.columns and df[COL_YEAR].notna().any():
    # 12.1 Representación 2D estable de DOCUMENTOS con TF-IDF + SVD (robusta y rápida)
    tfidf_all = TfidfVectorizer(max_df=0.9, min_df=MIN_DF_TEXT, ngram_range=(1,2))
    X_all = tfidf_all.fit_transform(df[COL_TEXT])
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    X_all_50 = svd.fit_transform(X_all)
    proj = PCA(n_components=2, random_state=SEED).fit_transform(X_all_50)  # 2D
    # Índices por periodo
    a_min, a_max = PERIOD_A
    b_min, b_max = PERIOD_B
    idx_A = df[COL_YEAR].between(a_min, a_max, inclusive="both").fillna(False).values
    idx_B = df[COL_YEAR].between(b_min, b_max, inclusive="both").fillna(False).values
    # Vocabulario de keywords (top por frecuencia global)
    kws_ranked = freq_kw_df.sort_values("count", ascending=False)["keyword"].tolist()[:SHIFT_TOP_KW]

    shift_rows = []
    plt.figure(figsize=(12,8))
    for kw in kws_ranked:
        # docs que contienen la keyword
        mask_kw = (X_kw[:, list(terms_kw).index(kw)].toarray().ravel() > 0) if kw in terms_kw else None
        if mask_kw is None: continue
        docs_A = np.where(mask_kw & idx_A)[0]
        docs_B = np.where(mask_kw & idx_B)[0]
        if len(docs_A) < SHIFT_MIN_DOCS_PER_PERIOD or len(docs_B) < SHIFT_MIN_DOCS_PER_PERIOD:
            continue
        # centroides en el espacio 2D (sobre documentos)
        cA = proj[docs_A].mean(axis=0)
        cB = proj[docs_B].mean(axis=0)
        # flecha
        plt.annotate("", xy=cB, xytext=cA, arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.7))
        # etiquetas coloreadas
        plt.text(cA[0], cA[1], f"{kw}", fontsize=8, color="green")
        plt.text(cB[0], cB[1], f"{kw}", fontsize=8, color="orange")
        # guardar medición (módulo del desplazamiento)
        dist = float(np.linalg.norm(cB - cA))
        shift_rows.append({"keyword": kw, "n_docs_A": len(docs_A), "n_docs_B": len(docs_B), "shift_2d": dist})

    if shift_rows:
        pd.DataFrame(shift_rows).sort_values("shift_2d", ascending=False) \
          .to_csv(Path(OUTDIR, "temporal_shift_keywords.csv"), index=False, encoding="utf-8")

    plt.title(f"Semantic change of keywords — {a_min}-{a_max} (green) → {b_min}-{b_max} (orange)")
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    savefig(Path(OUTDIR, "temporal_semantic_shift.png"))

print("\nLISTO ✅")
print(f"Carpeta de salida: {OUTDIR}")
