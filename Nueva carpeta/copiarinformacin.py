import os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== CONFIG ==================

INPUT = r"G:\Mi unidad\2025\Dra. Carmen Delgado\data\datawos_scopus_clean.csv"
OUTDIR = Path(r"G:\Mi unidad\2025\Dra. Carmen Delgado\data\outputs_classifier")
YEAR_MIN, YEAR_MAX = 2014, 2024

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
    # e-leadership family
    r"\be[-\s]?lead(er(ship)?|ing)\b", r"\bdigital leadership\b", r"\bvirtual leadership\b",
    r"\bonline leadership\b", r"\bremote leadership\b",
    # higher education
    r"\bhigher education\b", r"\buniversit(y|ies)\b", r"\bcollege(s)?\b",
    r"\btertiary education\b", r"\bhei(s)?\b",
    # teaching/learning modalities
    r"\bonline (teaching|learning)\b", r"\bremote (teaching|instruction)\b",
    r"\bdistance education\b", r"\bvirtual learning\b", r"\b(hybrid|blended|hyflex) learning\b",
    # outcomes (performance/wellbeing)
    r"\bteacher performance\b", r"\bfaculty performance\b", r"\bacademic performance\b", r"\bjob performance\b",
    r"\bwell[-\s]?being\b", r"\bwellbeing\b", r"\bpsychological wellbeing\b", r"\bstress\b", r"\bburnout\b",
    r"\bjob satisfaction\b", r"\bwork engagement\b", r"\bworkload\b"
]

# NEGATIVE_TERMS: common sources of off-topic noise
NEGATIVE_TERMS = [
    r"\b(se(vere)?\s*acute|influenza|covid-19|cancer|virus|clinical|patient|medical|nursing|surgery)\b",
    r"\b(agriculture|soil|crop|geology|mining|materials|physics|chemistry|biochemistry)\b",
    r"\b(environment(al)?|forestry|climate change|biodiversity)\b"
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
    # These act as “queries” describing your target topic
    "e-leadership in higher education for online and hybrid teaching",
    "digital leadership and its effects on teachers' performance and wellbeing",
    "virtual leadership practices supporting university faculty in remote learning environments",
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
