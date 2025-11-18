# -*- coding: utf-8 -*-
"""
02_eda_topics_embeddings_plus_v5.py  (EN labels)

- Year range filter (Biblioshiny-style).
- WordClouds, frequencies, LDA (+optional PyLDAvis).
- Embeddings (Sentence-Transformers) + UMAP/PCA + KMeans (filtered labels).
- Export top-10 per cluster (text & keywords).
- Robust, readable co-occurrence network (collaboration):
    * top-N keywords, edge threshold, max labels
    * optional split into two figures (half A/B) to reduce clutter
- Thematic Map (communities from co-occurrence).
- Readable dendrograms (truncated/top-K).
- Exports PNG and SVG.

Requires: numpy, pandas, matplotlib, scikit-learn
Optional: seaborn, sentence-transformers, umap-learn, scipy, squarify, pyLDAvis, networkx
"""

import os, ast, re, itertools, textwrap
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== KNOBS / QUICK SETTINGS ==============
INPUT  = r"G:\Mi unidad\2025\Master MIOSSOTTY KATHERINE NARANJO KEAN CHONG\articulo 2\data final\datawos_scopusbloque1_cleanfinalfinal.csv"
OUTDIR = str(Path(INPUT).with_name("eda_topics_embeddings_outputs"))
SEED   = 42

# Columns
COL_TITLE = "Title"
COL_TEXT  = "text_clean"
COL_YEAR  = "year"
COL_KWU   = "Keywords Unified"
COL_KWL   = "keywords_list"

# Year filter (Biblioshiny-like; ignored if no 'year')
YEAR_MIN  = 2014
YEAR_MAX  = 2024

# WordCloud / Top term bars
N_TOP_FREQ   = 25
MIN_DF_TEXT  = 5
MIN_DF_KW    = 1

# LDA
N_TOPICS_LDA = 6

# Embeddings / Clustering
KMEANS_K       = 6
UMAP_NN        = 15
UMAP_MIN_DIST  = 0.1
TOPK_SIM       = 5

# -------- Readable plots --------
# Keyword network (graph):
GRAPH_TOP_KEYWORDS   = 40        # draw these top keywords
GRAPH_MIN_EDGE_W     = 2         # co-occurrence threshold
LABEL_TOP_BY         = "degree"  # 'degree' or 'weight'
LABEL_TOP_NODES      = 25        # max number of labels in the graph
LABEL_MIN_NODE_FREQ  = 5         # don't label if node freq < this value
NODE_LABEL_WRAP      = 18        # characters per label line
GRAPH_SPLIT          = True      # also create two split figures (A/B)

# Dendrograms:
DENDRO_TOP_KW        = 120       # keywords to include (Top-K)
DENDRO_TRUNC_LEVELS  = 30        # truncation depth for documents dendrogram
KW_DENDRO_FONT       = 6         # label font size

# Thematic Map:
THEME_LABEL_TOPN     = 3
# ===================================

# Optional: seaborn
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import squarify
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

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

# ============== Utils ==============
os.makedirs(OUTDIR, exist_ok=True)

def savefig_all(path_noext, dpi=220):
    plt.tight_layout()
    png = str(Path(f"{path_noext}.png"))
    svg = str(Path(f"{path_noext}.svg"))
    plt.savefig(png, dpi=dpi)
    try:
        plt.savefig(svg)
    except Exception:
        pass
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
    if COL_KWL not in df.columns:
        df[COL_KWL] = df.get(COL_KWU, "").fillna("").apply(
            lambda s: str([t.strip() for t in str(s).split(";") if t.strip()])
        )
    return df

def wrap(s, width=NODE_LABEL_WRAP):
    return "\n".join(textwrap.wrap(str(s), width=width)) if s else s

# ============== 1) LOAD + YEAR FILTER ==============
df = pd.read_csv(INPUT, dtype=str, encoding="utf-8")
df = ensure_keywords_list(df)

if COL_TEXT not in df.columns:
    raise ValueError(f"Column {COL_TEXT} not found.")

df[COL_TEXT] = df[COL_TEXT].fillna("").astype(str)
df["kw_list_parsed"] = df[COL_KWL].apply(parse_kw_list)

if COL_YEAR in df.columns:
    df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
    df = df[(df[COL_YEAR] >= YEAR_MIN) & (df[COL_YEAR] <= YEAR_MAX) | df[COL_YEAR].isna()]
    df = df.reset_index(drop=True)

# Minimum cleaning
df[COL_TEXT] = df[COL_TEXT].str.replace(r"\bno\s+abstract\s+available\b", "", regex=True, flags=re.I)
df = df[df[COL_TEXT].str.split().str.len().fillna(0) >= 10].reset_index(drop=True)
print(f"Loaded records after filter: {len(df)}")

# ============== 2) WordClouds + Frequencies ==============
all_text = " ".join(df[COL_TEXT].tolist()).strip()
if all_text:
    wc = WordCloud(width=1600, height=800, background_color="white").generate(all_text)
    plt.figure(figsize=(14,7)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title("WordCloud — Title+Abstract (text_clean)")
    savefig_all(Path(OUTDIR, "wordcloud_title_abstract_text"))

all_kws = []
for kws in df["kw_list_parsed"]:
    all_kws.extend([k.lower() for k in kws if k])
kw_text = " ".join(all_kws).strip()
if kw_text:
    wc2 = WordCloud(width=1600, height=800, background_color="white").generate(kw_text)
    plt.figure(figsize=(14,7)); plt.imshow(wc2, interpolation="bilinear"); plt.axis("off")
    plt.title("WordCloud — Keywords (keywords_list)")
    savefig_all(Path(OUTDIR, "wordcloud_keywords"))

vec_text = CountVectorizer(max_df=0.8, min_df=MIN_DF_TEXT, ngram_range=(1,2))
X_text = vec_text.fit_transform(df[COL_TEXT])
terms_text = vec_text.get_feature_names_out()
freqs_text = np.asarray(X_text.sum(axis=0)).ravel()
freq_text_df = pd.DataFrame({"term": terms_text, "count": freqs_text}).sort_values("count", ascending=False)
freq_text_df.to_csv(Path(OUTDIR, "term_frequencies_text.csv"), index=False, encoding="utf-8")

top_text = freq_text_df.head(N_TOP_FREQ)
plt.figure(figsize=(12,6)); plt.barh(top_text["term"][::-1], top_text["count"][::-1])
plt.title("Top terms (Title+Abstract)")
savefig_all(Path(OUTDIR, "top_terms_text"))

from sklearn.feature_extraction.text import CountVectorizer as CV2
vec_kw = CV2(tokenizer=lambda s: s.split("|||"), lowercase=True, min_df=MIN_DF_KW)
kw_joined = ["|||".join([k.lower() for k in ks]) if ks else "" for ks in df["kw_list_parsed"]]
X_kw = vec_kw.fit_transform(kw_joined)
terms_kw = vec_kw.get_feature_names_out()
freqs_kw = np.asarray(X_kw.sum(axis=0)).ravel()
freq_kw_df = pd.DataFrame({"keyword": terms_kw, "count": freqs_kw}).sort_values("count", ascending=False)
freq_kw_df.to_csv(Path(OUTDIR, "keyword_frequencies.csv"), index=False, encoding="utf-8")

top_kw = freq_kw_df.head(N_TOP_FREQ)
plt.figure(figsize=(12,6)); plt.barh(top_kw["keyword"][::-1], top_kw["count"][::-1])
plt.title("Top Keywords (Index+Author)")
savefig_all(Path(OUTDIR, "top_keywords"))

# ============== 3) LDA ==============
lda = LatentDirichletAllocation(n_components=N_TOPICS_LDA, random_state=SEED, learning_method="batch")
lda.fit(X_text)

def top_terms_per_topic(model, feature_names, topn=10):
    out = []
    for _, comp in enumerate(model.components_):
        top_idx = comp.argsort()[-topn:][::-1]
        out.append([feature_names[i] for i in top_idx])
    return out

lda_topics = top_terms_per_topic(lda, terms_text, topn=10)
pd.DataFrame([{"topic": i+1, "top_terms": ", ".join(t)} for i, t in enumerate(lda_topics)]) \
  .to_csv(Path(OUTDIR, "lda_topics_top_terms.csv"), index=False, encoding="utf-8")

if 'PLDV_OK' in globals() and PLDV_OK:
    try:
        vis = pyLDAvis.sklearn.prepare(lda, X_text, vec_text, mds='tsne')
        pyLDAvis.save_html(vis, str(Path(OUTDIR, "lda_pyldavis.html")))
    except Exception:
        pass

# ============== 4) Embeddings + KMeans + Labels ==============
if 'ST_OK' in globals() and ST_OK:
    model_name = "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(model_name)
    docs = df[COL_TEXT].fillna("").tolist()
    doc_vecs = st_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    doc_vecs = np.asarray(doc_vecs)
else:
    tfidf_fallback = TfidfVectorizer(max_df=0.8, min_df=MIN_DF_TEXT, ngram_range=(1,2))
    doc_vecs = tfidf_fallback.fit_transform(df[COL_TEXT]).toarray()

# K diagnostics
ks, inertias, sils = [], [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=SEED, n_init="auto").fit(doc_vecs)
    ks.append(k); inertias.append(km.inertia_); sils.append(silhouette_score(doc_vecs, km.labels_))
pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": sils}).to_csv(Path(OUTDIR, "k_diagnostics.csv"), index=False)

# 2D
if HAS_UMAP:
    um = umap.UMAP(n_components=3, n_neighbors=UMAP_NN, min_dist=UMAP_MIN_DIST, random_state=SEED)
    doc_2d = um.fit_transform(doc_vecs)
else:
    doc_2d = PCA(n_components=2, random_state=SEED).fit_transform(doc_vecs)

kmeans = KMeans(n_clusters=KMEANS_K, random_state=SEED, n_init="auto")
labels = kmeans.fit_predict(doc_vecs)
df["cluster_kmeans"] = labels

# Cluster labels
tfidf_vec = TfidfVectorizer(max_df=0.85, min_df=3, ngram_range=(1,2))
X_tfidf = tfidf_vec.fit_transform(df[COL_TEXT])
terms = np.array(tfidf_vec.get_feature_names_out())

BLOCK = {"no","abstract","available","study","paper","result","results","approach","based","method","methods",
         "analysis","using","use","data","system","model","models","effect","effects","research","process",
         "application","applications","online","technology","digital","leader","leadership","virtual",
         "organization","organizational"}

Xkw_df = pd.DataFrame(X_kw.toarray(), columns=terms_kw)

cluster_text10, cluster_kw10, cluster_labels_rows = [], [], []
for c in sorted(np.unique(labels)):
    idx = np.where(labels == c)[0]
    sub = X_tfidf[idx]
    mean_tfidf = np.asarray(sub.mean(axis=0)).ravel()
    order = np.argsort(-mean_tfidf)
    top_terms = []
    for i in order:
        t = terms[i]
        if t in BLOCK or len(t) < 2 or t.isdigit(): continue
        top_terms.append(t)
        if len(top_terms) >= 10: break
    cluster_text10.append({"cluster": int(c), "top10_text": ", ".join(top_terms)})
    sub_kw = Xkw_df.iloc[idx]
    top_kw_c = sub_kw.mean(axis=0).sort_values(ascending=False).head(10).index.tolist()
    cluster_kw10.append({"cluster": int(c), "top10_keywords": ", ".join(top_kw_c)})
    short = ", ".join(top_terms[:3] + top_kw_c[:2])
    cluster_labels_rows.append({"cluster": int(c), "label": short})

pd.DataFrame(cluster_text10).to_csv(Path(OUTDIR, "cluster_top10_text.csv"), index=False, encoding="utf-8")
pd.DataFrame(cluster_kw10).to_csv(Path(OUTDIR, "cluster_top10_keywords.csv"), index=False, encoding="utf-8")
lab_map = {r["cluster"]: r["label"] for r in cluster_labels_rows}
pd.DataFrame(cluster_labels_rows).to_csv(Path(OUTDIR, "cluster_labels_short.csv"), index=False, encoding="utf-8")

# 2D scatter with labels
plt.figure(figsize=(10,8))
palette = (plt.cm.tab10.colors if not HAS_SNS else sns.color_palette("tab10", n_colors=len(np.unique(labels))))
for c in sorted(np.unique(labels)):
    idx = labels == c
    plt.scatter(doc_2d[idx,0], doc_2d[idx,1], s=12, color=palette[c % len(palette)], label=f"C{c}")
for c in sorted(np.unique(labels)):
    idx = labels == c
    cx, cy = doc_2d[idx,0].mean(), doc_2d[idx,1].mean()
    plt.text(cx, cy, wrap(lab_map.get(c, f"Cluster {c}"), 22), fontsize=9, weight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.7))
plt.legend(title="Clusters", loc="best", fontsize=8)
plt.title("UMAP/PCA + KMeans — Cluster labels (filtered)")
savefig_all(Path(OUTDIR, "umap_kmeans_labeled"))
df["cluster_label"] = [lab_map.get(c, f"Cluster {c}") for c in df["cluster_kmeans"]]
df.to_csv(Path(OUTDIR, "dataset_with_clusters.csv"), index=False, encoding="utf-8")

# ============== 5) Co-occurrence + Collaboration network ==============
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

def draw_keyword_graph(suffix=""):
    top_kw_ids = set(nodes_df.sort_values("weight", ascending=False).head(GRAPH_TOP_KEYWORDS)["id"])
    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        if r["id"] in top_kw_ids:
            G.add_node(r["id"], weight=int(r["weight"]))
    for _, r in edges_df.iterrows():
        a, b, w = r["source"], r["target"], int(r["weight"])
        if w >= GRAPH_MIN_EDGE_W and a in top_kw_ids and b in top_kw_ids:
            G.add_edge(a, b, weight=w)

    pos = nx.spring_layout(G, seed=SEED, k=0.35, iterations=150)

    deg = dict(G.degree())
    if LABEL_TOP_BY == "degree":
        ranking = sorted(G.nodes(), key=lambda n: deg.get(n,0), reverse=True)
    else:
        w = {n: G.nodes[n].get("weight", 0) for n in G.nodes()}
        ranking = sorted(G.nodes(), key=lambda n: w.get(n,0), reverse=True)
    to_label = [n for n in ranking if G.nodes[n].get("weight",0) >= LABEL_MIN_NODE_FREQ][:LABEL_TOP_NODES]
    labels_to_draw = {n: wrap(n) for n in to_label if n in pos}

    plt.figure(figsize=(14,11))
    node_sizes = [max(80, 25*G.nodes[n].get("weight",1)) for n in G.nodes()]
    edge_widths = [max(0.5, 0.6*G.edges[e]["weight"]) for e in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.22, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#4c78a8", alpha=0.85)
    nx.draw_networkx_labels(G, labels=labels_to_draw, pos=pos, font_size=9)
    plt.axis("off")
    plt.title(f"Collaboration network of keywords (top {GRAPH_TOP_KEYWORDS}, min edge={GRAPH_MIN_EDGE_W}){suffix}")
    savefig_all(Path(OUTDIR, f"keywords_collaboration_network{suffix}"))
    return G, pos

if HAS_NX and len(nodes_df) > 0:
    G, pos = draw_keyword_graph(suffix="")
    if GRAPH_SPLIT and len(G) > 0:
        xs = np.array([pos[n][0] for n in G.nodes()])
        thresh = np.median(xs)
        left_nodes  = [n for n in G.nodes() if pos[n][0] <= thresh]
        right_nodes = [n for n in G.nodes() if pos[n][0] >  thresh]

        sub = G.subgraph(left_nodes).copy()
        plt.figure(figsize=(14,11))
        node_sizes = [max(80, 25*sub.nodes[n].get("weight",1)) for n in sub.nodes()]
        edge_widths = [max(0.5, 0.6*sub.edges[e]["weight"]) for e in sub.edges()]
        nx.draw_networkx_edges(sub, pos, edgelist=sub.edges(), width=edge_widths, alpha=0.25, edge_color="gray")
        nx.draw_networkx_nodes(sub, pos, nodelist=sub.nodes(), node_size=node_sizes, node_color="#4c78a8", alpha=0.9)
        nx.draw_networkx_labels(sub, pos={n:pos[n] for n in sub.nodes()}, font_size=10)
        plt.axis("off"); plt.title("Collaboration network — Panel A (left)")
        savefig_all(Path(OUTDIR, "keywords_collaboration_network_A"))

        sub = G.subgraph(right_nodes).copy()
        plt.figure(figsize=(14,11))
        node_sizes = [max(80, 25*sub.nodes[n].get("weight",1)) for n in sub.nodes()]
        edge_widths = [max(0.5, 0.6*sub.edges[e]["weight"]) for e in sub.edges()]
        nx.draw_networkx_edges(sub, pos, edgelist=sub.edges(), width=edge_widths, alpha=0.25, edge_color="gray")
        nx.draw_networkx_nodes(sub, pos, nodelist=sub.nodes(), node_size=node_sizes, node_color="#4c78a8", alpha=0.9)
        nx.draw_networkx_labels(sub, pos={n:pos[n] for n in sub.nodes()}, font_size=10)
        plt.axis("off"); plt.title("Collaboration network — Panel B (right)")
        savefig_all(Path(OUTDIR, "keywords_collaboration_network_B"))

# ============== 6) Thematic Map ==============
if HAS_NX and len(nodes_df) > 0:
    if 'G' not in locals():
        top_kw_ids = set(nodes_df.sort_values("weight", ascending=False).head(GRAPH_TOP_KEYWORDS)["id"])
        G = nx.Graph()
        for _, r in nodes_df.iterrows():
            if r["id"] in top_kw_ids:
                G.add_node(r["id"], weight=int(r["weight"]))
        for _, r in edges_df.iterrows():
            a, b, w = r["source"], r["target"], int(r["weight"])
            if w >= GRAPH_MIN_EDGE_W and a in top_kw_ids and b in top_kw_ids:
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
        w_in = internal_weight(cset)
        density = 0.0 if n < 2 else 100.0 * (w_in / (n*(n-1)/2))
        cent = 10.0 * external_weight(cset)
        size = int(sum(nodes_c.get(k,0) for k in cset))
        top_label_kw = sorted(cset, key=lambda k: nodes_c.get(k,0), reverse=True)[:THEME_LABEL_TOPN]
        rows_tm.append({"theme_id": i, "label": ", ".join(top_label_kw), "density": density, "centrality": cent, "size": size})

    tm_df = pd.DataFrame(rows_tm)
    tm_df.to_csv(Path(OUTDIR, "thematic_map_stats.csv"), index=False, encoding="utf-8")

    cx = tm_df["centrality"].median() if not tm_df["centrality"].empty else 0
    cy = tm_df["density"].median() if not tm_df["density"].empty else 0

    plt.figure(figsize=(12,8))
    plt.axvline(cx, color="gray", linestyle="--", alpha=0.6)
    plt.axhline(cy, color="gray", linestyle="--", alpha=0.6)
    sizes = (tm_df["size"].fillna(1).astype(float) ** 0.85) * 10.0
    plt.scatter(tm_df["centrality"], tm_df["density"], s=sizes, alpha=0.75)

    for _, r in tm_df.iterrows():
        plt.text(r["centrality"], r["density"], wrap(r["label"], 24), fontsize=9, ha="center", va="center")

    xmin, xmax = plt.xlim(); ymin, ymax = plt.ylim()
    plt.text(xmin+(cx-xmin)*0.05, cy+(ymax-cy)*0.8, "Niche Themes", fontsize=10, color="gray")
    plt.text(cx+(xmax-cx)*0.6,   cy+(ymax-cy)*0.8, "Motor Themes", fontsize=10, color="gray")
    plt.text(xmin+(cx-xmin)*0.05,ymin+(cy-ymin)*0.1,"Emerging or\nDeclining Themes", fontsize=10, color="gray")
    plt.text(cx+(xmax-cx)*0.6,   ymin+(cy-ymin)*0.1,"Basic Themes", fontsize=10, color="gray")

    plt.xlabel("Relevance degree (Centrality)")
    plt.ylabel("Development degree (Density)")
    plt.title("Thematic Map (keywords communities)")
    savefig_all(Path(OUTDIR, "thematic_map"))

# ============== 7) Dendrograms (readable) ==============
if HAS_SCIPY:
    Z = linkage(pdist(doc_vecs, metric="cosine"), method="average")
    plt.figure(figsize=(18,6))
    dendrogram(Z, no_labels=True, count_sort=True, truncate_mode='level', p=DENDRO_TRUNC_LEVELS)
    plt.title(f"Document dendrogram (cosine, average) — truncated to {DENDRO_TRUNC_LEVELS} levels")
    savefig_all(Path(OUTDIR, "dendrogram_documents"))

    if len(terms_kw) > 1:
        keep = set(freq_kw_df.head(DENDRO_TOP_KW)["keyword"])
        cols = [i for i, term in enumerate(terms_kw) if term in keep]
        if len(cols) >= 2:
            Xkw_bin = (X_kw[:, cols] > 0).astype(int).toarray()
            labels_kw = [terms_kw[i] for i in cols]
            Zkw = linkage(pdist(Xkw_bin.T, metric="jaccard"), method="average")
            plt.figure(figsize=(20,6))
            dendrogram(Zkw, labels=labels_kw, leaf_rotation=90, leaf_font_size=KW_DENDRO_FONT)
            plt.title(f"Keyword dendrogram (1 - Jaccard) — Top {DENDRO_TOP_KW}")
            savefig_all(Path(OUTDIR, "dendrogram_keywords"))

# ============== 8) Cosine similarity (Top neighbors) ==============
dists = pairwise_distances(doc_vecs, metric="cosine")
sims  = 1.0 - dists
rows = []
for i in range(len(df)):
    idx = np.argsort(-sims[i])
    top = [j for j in idx if j != i][:TOPK_SIM]
    for j in top:
        rows.append({
            "doc_i": i, "doc_j": j,
            "sim_cosine": float(sims[i, j]),
            "title_i": (df.get(COL_TITLE, pd.Series([""])).iloc[i] if COL_TITLE in df.columns else ""),
            "title_j": (df.get(COL_TITLE, pd.Series([""])).iloc[j] if COL_TITLE in df.columns else "")
        })
pd.DataFrame(rows).to_csv(Path(OUTDIR, "doc_similarity_topK.csv"), index=False, encoding="utf-8")

print("\nDONE ✅")
print(f"Output → {OUTDIR}")
