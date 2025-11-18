# -*- coding: utf-8 -*-
"""
article_classifier_semantic_v2.py

Prioritizes papers for SLR with reportable rigor:
- Deduplicate by DOI or Title+Year
- Year filter
- Semantic score (embeddings) + dictionary score (regex for topic)
- Combined 'relevance_score' for triage
- OPTIONAL supervised model (only if a 'label' column exists with Relevant/Irrelevant):
    * TF-IDF + auxiliary features (embed_score, dict_score, relevance_score)
    * Train/test split (stratified)
    * Exports: ROC & PR curves, confusion matrix, metrics.csv, classification_report.txt

Exports:
- Excel: All_scores, Recommended, Borderline, Not_relevant
- CSV: article_prioritization.csv
- If supervised: supervised_report.txt, metrics.csv, roc_curve.png/svg, pr_curve.png/svg, confusion_matrix.png/svg

Requirements: pandas, numpy, scikit-learn, openpyxl, matplotlib
Optional: sentence-transformers (falls back to TF-IDF cosine if missing)
"""

import os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== CONFIG ==================

INPUT = r"G:\Mi unidad\2025\Master Italo Palacios\artículo version 2\data\datawos_scopus_abstract.csv"
OUTDIR = Path(r"G:\Mi unidad\2025\Master Italo Palacios\artículo version 2\data\outputs_classifier")
YEAR_MIN, YEAR_MAX = 2010, 2024

# Column names in your CSV (the code will try reasonable fallbacks)
COL_TITLE = "Title"
COL_ABS   = "Abstract"          # if missing, defaults to ""
COL_KW    = "Keywords Unified"  # can be empty
COL_YEAR  = "year"              # if your file uses "Year"/"PY", it's detected automatically
COL_DOI   = "DOI"
COL_LABEL = "label"             # expected values: 'Relevant' / 'Irrelevant' (case-insensitive)


# Nombres EXACTOS según tu CSV
COL_AUTHORS      = "Author full names"

COL_SOURCE       = "Source title"
COL_CITEDBY      = "Cited by"
COL_LINK         = "Link"
COL_AFFILS       = "Affiliations"
COL_AUTH_AFFILS  = "Authors with affiliations"
COL_LANG         = "Language of Original Document"
COL_DOCTYPE      = "Document Type"


# Thresholds for the triage buckets (based on relevance_score)
THRESH_RECOMMENDED = 0.55
THRESH_BORDERLINE  = 0.40

# Weights for the final score: relevance_score = W_EMBED * embed_score + W_DICT * dict_score
W_EMBED = 0.70
W_DICT  = 0.30

# Random seed for reproducibility
SEED = 42

# Plot quality
DPI_EXPORT = 240

os.makedirs(OUTDIR, exist_ok=True)

# ================== UTILITIES ==================

def col_first(df, names, default=None):
    """Return the first existing column among 'names', otherwise a Series(default)."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)

def normalize_text(s):
    """Lowercase + collapse spaces. Use for robust regex matching and TF-IDF."""
    s = (s or "")
    return re.sub(r"\s+", " ", s).strip().lower()

def deduplicate(df):
    """
    Deduplicate using:
      - DOI (if present), else
      - Title slice + '_' + Year
    """
    doi   = col_first(df, [COL_DOI, "doi"]).fillna("").astype(str).str.strip().str.lower()
    title = col_first(df, [COL_TITLE, "title"]).fillna("").astype(str).str.strip().str.lower()
    year  = col_first(df, [COL_YEAR, "Year", "PY"]).apply(pd.to_numeric, errors="coerce").astype("Int64")
    key   = np.where(doi.values != "", doi.values, (title.str.slice(0,120) + "_" + year.astype(str)).values)
    return df.assign(_dup_key=key).loc[lambda x: ~x.duplicated("_dup_key")].drop(columns=["_dup_key"])

def build_combined_text(row):
    """
    Combine Title + Abstract + Keywords Unified into a single text string
    that we will feed into embeddings / TF-IDF and regex dictionaries.
    """
    pieces = []
    for c in [COL_TITLE, COL_ABS, COL_KW]:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            pieces.append(str(row[c]))
    return " ".join(pieces)

def minmax(x):
    """Rescale a 1-D array to [0, 1]."""
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x) if np.ptp(x) == 0 else (x - x.min()) / (x.max() - x.min())

# ================== LOAD & PREP ==================

df0 = pd.read_csv(INPUT, dtype=str, encoding="utf-8")

# Year filtering
year_series = col_first(df0, [COL_YEAR, "Year", "PY"]).apply(pd.to_numeric, errors="coerce")
if year_series.notna().any():
    df0 = df0[(year_series.between(YEAR_MIN, YEAR_MAX)) | (year_series.isna())].copy()

# Deduplicate
df = deduplicate(df0).reset_index(drop=True)

# Combined text
df["combined_text"] = df.apply(build_combined_text, axis=1).fillna("")
df["combined_text_norm"] = df["combined_text"].apply(normalize_text)

print(f"Loaded after year-filter & dedup: {len(df)} records")

# ================== DICTIONARY SCORING ==================
# POSITIVE_TERMS: core topic signals aligned to your scope
POSITIVE_TERMS = [
  # Contexto urbano/institucional + arbolado
    r"\burban (forest(ry)?|tree(s)?|greening|green infrastructure|park(s)?)\b",
    r"\b(institutional|campus|university|school|hospital|municipal|government|corporate)\s+(tree(s)?|greening|landscape|planting)\b",
    r"\b(tree[-\s]?planting|afforestation|reforestation|ecosystem restoration|ecological restoration|plantation forestry|silviculture)\b",
    r"\bstreet trees?\b", r"\burban tree canopy\b", r"\bcanopy (cover|coverage|mapping)\b",

    # Métricas / impactos ambientales
    r"\b(NDVI|normalized difference vegetation index)\b",
    r"\b(LST|land surface temperature|surface temperature)\b",
    r"\b(urban heat island|UHI|thermal comfort|microclimate)\b",
    r"\b(air quality|PM2\.?5|PM10|particulate matter)\b",
    r"\b(carbon (sequestration|storage|stock(s)?|flux)|CO2|carbon footprint)\b",
    r"\b(biodiversity|species (richness|diversity))\b",
    r"\b(stormwater|runoff|green stormwater infrastructure|GSI|watershed management)\b",

    # RS/GIS y herramientas
    r"\b(remote sensing|satellite (imagery|data)|Sentinel[-\s]?2|Landsat|GIS|geographic information system(s)?|i[-\s]?tree( eco)?)\b",

    # Implementación / gestión / gobernanza
    r"\b(governance|policy|regulation|stakeholder(s)?|participation|community engagement)\b",
    r"\b(payments?\s+for\s+ecosystem\s+services|PES|funding|finance|incentive(s)?)\b",
    r"\b(maintenance|monitoring|survival (rate)?|MRV|audit(s)?)\b",
]

# NEGATIVE_TERMS: common sources of off-topic noise
NEGATIVE_TERMS = [
    # Medicina clínica dura (fuera de salud ambiental/comunitaria)
    r"\b(se(vere)?\s*acute|oncolog(y|ical)|cancer|tumou?r|virus|clinical|patient|surgery|nursing|ICU)\b",
    # Dominio marino / pesquero
    r"\b(marine|ocean(ic)?|offshore|coral|fishery|fisheries)\b",
    # Agro productivo lejos de ciudad (mantén suelo/agua porque sí te sirven en urbano)
    r"\b(crop(s)?|agronom(y|ic)|pasture|rangeland|farmland)\b",
    # Química de materiales pura
    r"\b(semiconductor|microelectronic(s)?|crystallograph(y|ic)|photovoltaic(s)?)\b",
    # Tu ecuación excluye “invasive species”: penalizamos alineado
    r"\b(invasive species|biological invasion)\b",
]

POS_RX = [re.compile(p, flags=re.I) for p in POSITIVE_TERMS]
NEG_RX = [re.compile(p, flags=re.I) for p in NEGATIVE_TERMS]

def dict_score(text):
    """
    Count positive matches and penalize with negatives.
    Map to [0,1] using 1-exp(-raw) for a smooth saturation curve.

    Interpreting dict_score:
      ~0.0  -> very few positive signals or strong negatives
      ~0.3  -> some topical hints
      ~0.6+ -> strong presence of core-topic terms
      ~0.85 -> very dense in focus vocabulary
    """
    pos = sum(bool(rx.search(text)) for rx in POS_RX)
    neg = sum(bool(rx.search(text)) for rx in NEG_RX)
    raw = max(0, pos - 0.5 * neg)
    return 1 - np.exp(-raw)

df["dict_score"] = df["combined_text"].fillna("").apply(dict_score)

# ================== SEMANTIC SIMILARITY (EMBEDDINGS) ==================

reference_texts = [
    "Urban and institutional tree planting initiatives between 2010 and 2024, combining bibliometric and systematic perspectives",
    "Comparative analysis of urban forestry and green infrastructure programs in universities and institutional environments",
    "Bibliometric and systematic synthesis of tree planting projects and their environmental and governance outcomes",
    "Urban greening and tree restoration programs addressing sustainability, resilience, and climate adaptation in institutional settings"

]

USE_ST = True
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    USE_ST = False

if USE_ST:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_emb = model.encode(reference_texts, normalize_embeddings=True)
    art_emb = model.encode(df["combined_text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    # cosine similarity of each article vs all references (average across references)
    sims = util.cos_sim(art_emb, ref_emb).mean(dim=1).cpu().numpy()
else:
    # Fallback: TF-IDF cosine vs a concatenated reference query
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    q = " ".join(reference_texts)
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))
    X = vectorizer.fit_transform([q] + df["combined_text"].tolist())
    sims = cosine_similarity(X[1:], X[0]).ravel()

df["embed_score"] = minmax(sims)

# ================== FINAL TRIAGE SCORE ==================

df["relevance_score"] = W_EMBED * df["embed_score"] + W_DICT * df["dict_score"]

def decide_bucket(score):
    """
    Buckets for quick triage:
      Recommended: relevance_score >= THRESH_RECOMMENDED
      Borderline : THRESH_BORDERLINE <= score < THRESH_RECOMMENDED
      Not_relevant: otherwise
    """
    if score >= THRESH_RECOMMENDED:
        return "Recommended"
    elif score >= THRESH_BORDERLINE:
        return "Borderline"
    else:
        return "Not_relevant"

df["bucket"] = df["relevance_score"].apply(decide_bucket)

# Columns to export (keep common bibliographic fields first)
order_cols = [c for c in [ COL_TITLE, COL_DOI, COL_YEAR, COL_SOURCE, COL_CITEDBY, COL_LANG, COL_DOCTYPE,
    COL_AUTHORS, COL_AFFILS, COL_AUTH_AFFILS, COL_LINK, COL_ABS, COL_KW] if c in df.columns]
score_cols = ["embed_score", "dict_score", "relevance_score", "supervised_prob", "bucket"]
export_cols = order_cols + score_cols

# Initialize supervised_prob (filled later if we train)
df["supervised_prob"] = np.nan

# ================== SUPERVISED MODEL (OPTIONAL) ==================
# Only runs if your dataset has a 'label' column with values 'Relevant'/'Irrelevant'
report_txt = ""
has_labels = (COL_LABEL in df.columns) and df[COL_LABEL].notna().any()

if has_labels:
    # --- Prepare y (labels) ---
    y_raw = df[COL_LABEL].fillna("").astype(str).str.strip().str.lower()
    # Allow some flexible mapping (you can extend these if needed)
    map_dict = {
        "relevant": 1, "relevance": 1, "pos": 1, "positive": 1, "1": 1,
        "irrelevant": 0, "not_relevant": 0, "neg": 0, "negative": 0, "0": 0
    }
    y = y_raw.map(map_dict)
    mask = y.isin([0, 1])

    if mask.sum() >= 20:  # need enough labels to split
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            classification_report, roc_auc_score, roc_curve,
            precision_recall_curve, average_precision_score,
            confusion_matrix
        )
        from scipy.sparse import hstack

        # Features: TF-IDF of combined_text + auxiliary scores
        X_text = df.loc[mask, "combined_text"].fillna("")
        y_use  = y[mask]

        vect = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))
        X_tfidf = vect.fit_transform(X_text)

        aux_feats = df.loc[mask, ["embed_score", "dict_score", "relevance_score"]].values
        X_full = hstack([X_tfidf, aux_feats])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_use, test_size=0.2, random_state=SEED, stratify=y_use
        )

        clf = LogisticRegression(max_iter=2000, n_jobs=None)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # --- Metrics ---
        auc = roc_auc_score(y_te, y_prob)
        rep = classification_report(y_te, y_pred, digits=3)
        ap  = average_precision_score(y_te, y_prob)

        # Save text report
        report_txt = (
            f"=== Supervised model: Logistic Regression (TF-IDF + aux features) ===\n"
            f"AUC (ROC): {auc:.3f}\n"
            f"Average Precision (PR AUC): {ap:.3f}\n"
            f"{rep}\n"
        )
        print(report_txt)
        with open(OUTDIR / "supervised_report.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)

        # Save metrics CSV
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        metrics_row = pd.DataFrame([{
            "AUC_ROC": auc,
            "Average_Precision": ap,
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, zero_division=0),
            "Recall": recall_score(y_te, y_pred, zero_division=0),
            "F1": f1_score(y_te, y_pred, zero_division=0),
            "n_train": int(y_tr.shape[0]),
            "n_test": int(y_te.shape[0]),
            "pos_rate_test": float(y_te.mean())
        }])
        metrics_row.to_csv(OUTDIR / "metrics.csv", index=False, encoding="utf-8")

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        plt.figure(figsize=(6.5,5))
        plt.plot(fpr, tpr, lw=2, label=f"LogReg (AUC = {auc:.3f})")
        plt.plot([0,1],[0,1], lw=1, linestyle="--", color="grey", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve – Article Relevance Classifier")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUTDIR / "roc_curve.png", dpi=DPI_EXPORT)
        try: plt.savefig(OUTDIR / "roc_curve.svg")
        except Exception: pass
        plt.close()

        # --- Precision–Recall Curve ---
        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        plt.figure(figsize=(6.5,5))
        plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve – Article Relevance Classifier")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(OUTDIR / "pr_curve.png", dpi=DPI_EXPORT)
        try: plt.savefig(OUTDIR / "pr_curve.svg")
        except Exception: pass
        plt.close()

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_te, y_pred, labels=[0,1])
        fig = plt.figure(figsize=(5.5,5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Irrelevant","Relevant"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["Irrelevant","Relevant"])
        for (i,j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center", fontsize=11)
        ax.set_title("Confusion Matrix – Test Set")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(OUTDIR / "confusion_matrix.png", dpi=DPI_EXPORT)
        try: plt.savefig(OUTDIR / "confusion_matrix.svg")
        except Exception: pass
        plt.close()

        # Probabilities for the labeled subset (optional enrichment)
        df.loc[mask, "supervised_prob"] = clf.predict_proba(X_full)[:, 1]
    else:
        print("⚠ Not enough labeled rows to train a supervised model (need ≥ 20). Skipping supervised block.")

# ================== EXPORTS ==================

x_path = OUTDIR / "article_prioritization.xlsx"
with pd.ExcelWriter(x_path, engine="openpyxl", mode="w") as w:
    df.sort_values("relevance_score", ascending=False)[export_cols].to_excel(
        w, sheet_name="All_scores", index=False
    )
    for name in ["Recommended", "Borderline", "Not_relevant"]:
        df[df["bucket"] == name].sort_values("relevance_score", ascending=False)[export_cols] \
          .to_excel(w, sheet_name=name, index=False)

df.sort_values("relevance_score", ascending=False)[export_cols] \
  .to_csv(OUTDIR / "article_prioritization.csv", index=False, encoding="utf-8")

# PRISMA-like counts for console
counts = df["bucket"].value_counts().to_dict()
summary = {
    "total_after_dedup": int(len(df)),
    "recommended": int(counts.get("Recommended", 0)),
    "borderline": int(counts.get("Borderline", 0)),
    "not_relevant": int(counts.get("Not_relevant", 0)),
}
print("\nSummary:", summary)

if len(report_txt) > 0:
    print("Saved: supervised_report.txt, metrics.csv, roc_curve.png/svg, pr_curve.png/svg, confusion_matrix.png/svg")

print(f"\n✅ Done. Files in: {OUTDIR}")
print(f"   - {x_path.name}")
print("   - article_prioritization.csv")

# ================== SHORTLIST AJUSTADA (descarga) ==================

def _count_matches(text, patterns):
    return sum(bool(rx.search(text)) for rx in patterns)

df["pos_hits"] = df["combined_text"].fillna("").apply(lambda s: _count_matches(s, POS_RX))
df["neg_hits"] = df["combined_text"].fillna("").apply(lambda s: _count_matches(s, NEG_RX))

# UMBRALES DE PUNTAJE (ajústalos según el volumen que quieras descargar)
MIN_EMBED      = 0.55   # similitud semántica mínima
MIN_DICT       = 0.35   # fuerza diccionario mínima
MIN_POS_HITS   = 2      # mínimo de señales positivas
ONLY_RECOMMEND = True   # exigir además bucket == "Recommended"

mask_core = (
    (df["embed_score"] >= MIN_EMBED) &
    (df["dict_score"]  >= MIN_DICT) &
    (df["pos_hits"]    >= MIN_POS_HITS)
)
if ONLY_RECOMMEND:
    mask_core &= (df["bucket"] == "Recommended")

df["recommend_download"] = mask_core
df["download_priority"]  = np.where(df["recommend_download"], df["relevance_score"], np.nan)

df["why"] = (
    "bucket=" + df["bucket"].astype(str) +
    "; embed=" + df["embed_score"].round(2).astype(str) +
    "; dict=" + df["dict_score"].round(2).astype(str) +
    "; pos_hits=" + df["pos_hits"].astype(str) +
    "; neg_hits=" + df["neg_hits"].astype(str)
)

print("\n=== Audit shortlist ===")
print(df["bucket"].value_counts())
print("Recomendados (descargar):", int(df["recommend_download"].sum()))

# ================== ETIQUETAS TEMÁTICAS (labels) ==================
# Define patrones por etiqueta (puedes ampliar o ajustar a tu gusto)
LABELS = {
    "campus_institucional": [
        r"\b(campus|university|universities|polytechnic|institute|school|faculty|hospital|government|municipal|corporate)\b",
        r"\b(institutional|campus)\s+(tree|trees|planting|greening|landscape)\b",
    ],
    "urban_tree": [
        r"\burban (forest(ry)?|tree(s)?|greening|green infrastructure|park(s)?)\b",
        r"\bstreet trees?\b", r"\burban tree canopy\b", r"\bcanopy (cover|coverage|mapping)\b",
        r"\btree[-\s]?planting\b|\b(afforestation|reforestation|ecosystem restoration|ecological restoration)\b",
    ],
    "rs_gis": [
        r"\b(remote sensing|satellite (imagery|data)|Sentinel[-\s]?2|Landsat|GIS|geographic information system(s)?|i[-\s]?tree( eco)?)\b",
    ],
    "metrics_heat_uhi": [
        r"\b(NDVI|normalized difference vegetation index)\b",
        r"\b(LST|land surface temperature|surface temperature)\b",
        r"\b(urban heat island|UHI|thermal comfort|microclimate)\b",
        r"\b(air quality|PM2\.?5|PM10|particulate matter)\b",
    ],
    "carbon_biodiversity": [
        r"\b(carbon (sequestration|storage|stock(s)?|flux)|CO2|carbon footprint)\b",
        r"\b(biodiversity|species (richness|diversity))\b",
    ],
    "water_stormwater": [
        r"\b(stormwater|runoff|green stormwater infrastructure|GSI|watershed management)\b",
    ],
    "governance_finance": [
        r"\b(governance|policy|regulation|stakeholder(s)?|participation|community engagement)\b",
        r"\b(payments?\s+for\s+ecosystem\s+services|PES|funding|finance|incentive(s)?)\b",
        r"\b(maintenance|monitoring|survival (rate)?|MRV|audit(s)?)\b",
    ],
}

# Compila los patrones (una vez)
LABELS_RX = {k: [re.compile(p, flags=re.I) for p in pats] for k, pats in LABELS.items()}

def match_labels(text, labels_rx=LABELS_RX):
    text = text or ""
    matched = []
    for label, rx_list in labels_rx.items():
        if any(rx.search(text) for rx in rx_list):
            matched.append(label)
    return matched

df["labels"]    = df["combined_text"].apply(match_labels)
df["n_labels"]  = df["labels"].apply(len)
df["labels_str"] = df["labels"].apply(lambda xs: "; ".join(xs) if xs else "")

# === CONFIGURACIÓN DE FILTRADO POR ETIQUETAS ===
# Selecciona las etiquetas que QUIERES en la shortlist final (OR/AND)
SELECT_LABELS = ["urban_tree", "campus_institucional"]  # <-- ajusta a tu gusto
LABEL_MODE_AND = False  # False = OR (al menos una), True = AND (todas)

# Opcional: etiquetas a excluir
EXCLUDE_LABELS = []  # p.ej. ["water_stormwater"]

def keep_by_selected(labels_found, include=SELECT_LABELS, mode_and=LABEL_MODE_AND):
    if not include:
        return True  # si no seleccionas nada, no restringe por etiquetas
    s = set(labels_found)
    inc = set(include)
    return inc.issubset(s) if mode_and else len(s.intersection(inc)) > 0

def has_excluded(labels_found, exclude=EXCLUDE_LABELS):
    if not exclude:
        return False
    return len(set(labels_found).intersection(set(exclude))) > 0

df["label_include_ok"] = df["labels"].apply(lambda ls: keep_by_selected(ls, SELECT_LABELS, LABEL_MODE_AND))
df["label_exclude_hit"] = df["labels"].apply(lambda ls: has_excluded(ls, EXCLUDE_LABELS))

# Shortlist combinando score + etiquetas
df["recommend_download_by_tags"] = (
    df["recommend_download"] &
    df["label_include_ok"] &
    (~df["label_exclude_hit"])
)

print("Shortlist por etiquetas (recomendados + labels):", int(df["recommend_download_by_tags"].sum()))

# ================== EXPORTS (con shortlist + etiquetas) ==================

x_path = OUTDIR / "article_prioritization.xlsx"

order_cols = [c for c in [ COL_TITLE, COL_DOI, COL_YEAR, COL_SOURCE, COL_CITEDBY, COL_LANG, COL_DOCTYPE,
    COL_AUTHORS, COL_AFFILS, COL_AUTH_AFFILS, COL_LINK, COL_ABS, COL_KW] if c in df.columns]

score_cols = [
    "embed_score", "dict_score", "pos_hits", "neg_hits",
    "relevance_score", "bucket", "recommend_download",
    "download_priority", "why",
    "labels_str", "n_labels", "label_include_ok", "label_exclude_hit",
    "recommend_download_by_tags", "supervised_prob"
]

export_cols = order_cols + [c for c in score_cols if c not in order_cols]

sort_cols = ["recommend_download_by_tags", "recommend_download", "download_priority", "relevance_score"]
ascending = [False, False, False, False]
out = df.sort_values(sort_cols, ascending=ascending)

with pd.ExcelWriter(x_path, engine="openpyxl", mode="w") as w:
    # Todo
    out[export_cols].to_excel(w, sheet_name="All_scores", index=False)
    # Shortlist original por score
    out[out["recommend_download"]][export_cols].to_excel(w, sheet_name="Shortlist (Download)", index=False)
    # Shortlist filtrada por etiquetas + score
    out[out["recommend_download_by_tags"]][export_cols].to_excel(w, sheet_name="Shortlist (Tags+Download)", index=False)
    # Los que no pasan por etiquetas (pero sí tenían buen score)
    hold_mask = (out["recommend_download"]) & (~out["recommend_download_by_tags"])
    out[hold_mask][export_cols].to_excel(w, sheet_name="Hold (Score OK, sin tags)", index=False)
    # Buckets
    for name in ["Recommended", "Borderline", "Not_relevant"]:
        out[out["bucket"] == name][export_cols].to_excel(w, sheet_name=name, index=False)

out[export_cols].to_csv(OUTDIR / "article_prioritization.csv", index=False, encoding="utf-8")

counts = df["bucket"].value_counts().to_dict()
summary = {
    "total_after_dedup": int(len(df)),
    "recommended_bucket": int(counts.get("Recommended", 0)),
    "borderline_bucket": int(counts.get("Borderline", 0)),
    "not_relevant_bucket": int(counts.get("Not_relevant", 0)),
    "shortlist_download_true": int(df["recommend_download"].sum()),
    "shortlist_download_by_tags_true": int(df["recommend_download_by_tags"].sum()),
}
print("\nSummary:", summary)

print(f"\n✅ Done. Files in: {OUTDIR}")
print(f"   - {x_path.name}")
print("   - article_prioritization.csv")

