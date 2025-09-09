# -*- coding: utf-8 -*-
# ============================================================
# 09_modelos_robustos.py  (versión corregida QWK)
# CV (5-fold) + Hold-out con RobustScaler
# Modelos: RF, SVM (RBF), XGB (si está instalado), MLP, LogReg, KNN
# Métricas: accuracy, balanced_accuracy, f1_macro, QWK
# Plots: Matriz de Confusión + Curvas ROC (OvR, micro, macro)
# Salidas: CSV de CV, CSV de holdout, PNGs por modelo
# ============================================================
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score,
    roc_auc_score, roc_curve, auc, make_scorer
)
from sklearn.preprocessing import RobustScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline

# Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# XGBoost (opcional)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- RUTAS -----------------
BASE = r"G:/Mi unidad/2025/master Ms. JOHANA VERONICA ESPINEL GUADALU"
DATA_XLSX = os.path.join(BASE, "data", "datos_codificados.xlsx")
OUT_DIR   = os.path.join(BASE, "resultados_modelos_robustos")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CV_CSV       = os.path.join(OUT_DIR, "09_modelos_robustos_cv.csv")
OUT_HOLDOUT_CSV  = os.path.join(OUT_DIR, "09_modelos_robustos_holdout.csv")
OUT_BEST_JSON    = os.path.join(OUT_DIR, "09_mejor_por_qwk.json")

# ----------------- CARGA -----------------
df = pd.read_excel(DATA_XLSX)

# ----------------- VARIABLES (ajústalas) -----------------
target = "V27"
features = ["V111","V112","V113","V114","V115"]

X_raw = df[features].copy()
y_raw = df[target].copy()

# Etiquetas a 0..K-1 (y mostrar originales en reportes/plots)
le = LabelEncoder()
y = le.fit_transform(y_raw.values)
class_labels = le.classes_  # p.ej. array([1,2,3,4,5])

# ----------------- CV / Scorers -----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

scorers = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "qwk": make_scorer(qwk)   # <<< CORREGIDO
}

# ----------------- Modelos -----------------
def build_models(n_classes):
    modelos = {}

    modelos["RF"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ))
    ])

    modelos["SVM_rbf"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

    modelos["LogReg"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])

    modelos["KNN"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
            metric="minkowski"
        ))
    ])

    modelos["MLP"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42
        ))
    ])

    if HAS_XGB:
        modelos["XGB"] = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="auto",
                random_state=42,
                n_jobs=-1
            ))
        ])

    return modelos

# ----------------- Utilidades de Plots -----------------
def plot_confusion(y_true, y_pred, model_name, out_dir, display_labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{model_name} - Matriz de Confusión (Hold-out)")
    out_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_roc_multiclass(y_true, y_proba, model_name, out_dir, display_labels):
    classes = np.arange(y_proba.shape[1])
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = y_bin.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=(8, 7))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Clase {display_labels[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["micro"], tpr["micro"], lw=2, ls="--", label=f"micro (AUC={roc_auc['micro']:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], lw=2, ls="--", label=f"macro (AUC={roc_auc['macro']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{model_name} - Curvas ROC (Hold-out, OvR)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_roc_curves.png")
    plt.savefig(out_path, dpi=150); plt.close(fig)
    return out_path, roc_auc["macro"], roc_auc["micro"]

# ----------------- MAIN -----------------
if __name__ == "__main__":
    modelos = build_models(n_classes=len(np.unique(y)))

    # ===== CV (5-fold) =====
    cv_rows = []
    for name, pipe in modelos.items():
        print(f"\n▶ CV 5-fold → {name}")
        cv_out = cross_validate(pipe, X_raw, y, scoring=scorers, cv=cv, n_jobs=-1, return_train_score=False)
        row = {
            "model": name,
            "acc_mean":  np.mean(cv_out["test_accuracy"]),
            "acc_std":   np.std(cv_out["test_accuracy"]),
            "bacc_mean": np.mean(cv_out["test_balanced_accuracy"]),
            "bacc_std":  np.std(cv_out["test_balanced_accuracy"]),
            "f1_macro_mean": np.mean(cv_out["test_f1_macro"]),
            "f1_macro_std":  np.std(cv_out["test_f1_macro"]),
            "qwk_mean":  np.mean(cv_out["test_qwk"]),   # <<< CORREGIDO
            "qwk_std":   np.std(cv_out["test_qwk"]),
            "folds": cv.get_n_splits(),
            "status": "ok"
        }
        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows)
    cv_df = cv_df.sort_values(by=["qwk_mean","bacc_mean","f1_macro_mean"], ascending=False)
    cv_df.to_csv(OUT_CV_CSV, index=False, encoding="utf-8")
    print("\n== CV (5-fold) - TOP por QWK ==")
    print(cv_df.head(10))
    with open(OUT_BEST_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_by_qwk": cv_df.iloc[0]["model"]}, f, ensure_ascii=False, indent=2)

    # ===== HOLD-OUT 20% =====
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )

    hold_rows = []
    print("\n== Hold-out (20%) por modelo ==")
    for name, pipe in modelos.items():
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)
            auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr")
        else:
            y_proba = None
            auc_ovr = np.nan

        acc  = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1m  = f1_score(y_test, y_pred, average="macro")
        kq   = cohen_kappa_score(y_test, y_pred, weights="quadratic")

        # print ordenado
        print(f"{name:10s} | acc={acc:.3f} | bacc={bacc:.3f} | f1={f1m:.3f} | QWK={kq:.3f} | AUCovr={auc_ovr if np.isnan(auc_ovr) else round(auc_ovr,3)}")
        print(classification_report(
            y_test, y_pred,
            labels=np.arange(len(class_labels)),
            target_names=[str(c) for c in class_labels]
        ))

        cm_path = plot_confusion(y_test, y_pred, name, OUT_DIR, display_labels=class_labels)
        roc_path, auc_macro, auc_micro = (None, np.nan, np.nan)
        if y_proba is not None:
            roc_path, auc_macro, auc_micro = plot_roc_multiclass(y_test, y_proba, name, OUT_DIR, class_labels)

        hold_rows.append({
            "model": name,
            "acc": acc, "bacc": bacc, "f1_macro": f1m, "qwk": kq,
            "auc_ovr": auc_ovr,
            "auc_macro": auc_macro, "auc_micro": auc_micro,
            "confusion_png": cm_path,
            "roc_png": roc_path
        })

    hold_df = pd.DataFrame(hold_rows).sort_values(by=["qwk","bacc","f1_macro"], ascending=False)
    hold_df.to_csv(OUT_HOLDOUT_CSV, index=False, encoding="utf-8")

    print("\n== Resumen HOLD-OUT (ordenado por QWK) ==")
    print(hold_df[["model","acc","bacc","f1_macro","qwk","auc_ovr"]])

    print("\n✅ Archivos guardados:")
    print(" - CV      :", OUT_CV_CSV)
    print(" - Hold-out:", OUT_HOLDOUT_CSV)
    print(" - Best JSON:", OUT_BEST_JSON)
    print(" - PNGs por modelo en:", OUT_DIR)
