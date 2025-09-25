"""
Train and evaluate a Logistic Regression baseline classifier for hate/offensive speech detection.

Inputs
------
- labeled_data_clean.csv (from 01_preprocess.py) with columns:
    text  : preprocessed tweet text
    label : integer class (0=hate, 1=offensive, 2=neither)

Outputs (written to OUTPUT/ by default)
-------
- logistic_classification_report.txt   : full precision/recall/F1 by class
- logistic_metrics.json                : macro/micro/weighted metrics + accuracy
- logistic_best_params.json            : best hyperparameters from grid search
- confusion_matrix_logistic.png        : confusion matrix heatmap
- model_logistic.joblib                : fitted classifier (Pipeline)

Usage
-----
python SCRIPTS/02_model_logistic.py --input DATA/processed/labeled_data_clean.csv --out_dir OUTPUT --test_size 0.2 --seed 42

Notes
-----
- Uses stratified train/test split to preserve class balance.
- Class imbalance is addressed via class_weight='balanced' in LogisticRegression.
- Grid search tunes C and n-gram ranges; you can extend the grid as needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
import joblib


def load_data(path: Path):
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'text' and 'label' columns. Did you run 01_preprocess.py?")
    # Drop any missing rows just in case
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def build_pipeline():
    # Vectorizer + Logistic Regression in a Pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode")),
        ("clf", LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    return pipe


def get_param_grid():
    # Modest grid for quick runs; expand if you have more time/compute
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "tfidf__max_df": [0.9, 1.0],
        "clf__C": [0.1, 1.0, 3.0, 10.0],
        # "tfidf__analyzer": ["word", "char"],
        # # For words, (1,2) is typical; for chars, (3,5) or (3,6) work well on tweets
        # "tfidf__ngram_range": [(1, 2), (3, 5)],
        # "tfidf__sublinear_tf": [True],
        # "tfidf__min_df": [1, 2],
        # "tfidf__max_df": [0.85, 0.9, 1.0],
        # "clf__C": [0.5, 1.0, 3.0, 10.0],
    }
    return grid


def evaluate_and_save(y_true, y_pred, labels, out_dir: Path):
    # Text report
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    (out_dir / "logistic_classification_report.txt").write_text(report, encoding="utf-8")

    # Metrics JSON
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
    }
    (out_dir / "logistic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)  # no explicit colors
    ax.set_title("Confusion Matrix — Logistic Regression")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_logistic.png", dpi=300)
    plt.close(fig)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Logistic Regression baseline on TF-IDF features.")
    parser.add_argument("--input", type=str, required=True, help="Path to DATA/processed/labeled_data_clean.csv (from 01_preprocess.py).")
    parser.add_argument("--out_dir", type=str, default="OUTPUT", help="Directory to write outputs.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction for train_test_split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv_folds", type=int, default=5, help="StratifiedKFold splits for GridSearchCV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = load_data(input_path)
    X = df["text"].astype(str).values
    y = df["label"].values
    labels_sorted = sorted(np.unique(y).tolist())  # typically [0, 1, 2]

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 3) Build pipeline + grid
    pipe = build_pipeline()
    param_grid = get_param_grid()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # 4) Grid search
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=3,
        scoring="f1_macro",
        refit=True,
    )
    gs.fit(X_train, y_train)

    # Save best params
    best_params = gs.best_params_
    (out_dir / "logistic_best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    # 5) Evaluate on test
    y_pred = gs.predict(X_test)
    metrics = evaluate_and_save(y_test, y_pred, labels_sorted, out_dir)

    # 6) Persist model (pipeline)
    joblib.dump(gs.best_estimator_, out_dir / "model_logistic.joblib")

    # Console summary
    print("[INFO] Finished Logistic Regression baseline.")
    print(f"[INFO] Test metrics: {json.dumps(metrics, indent=2)}")
    print(f"[INFO] Best params: {json.dumps(best_params, indent=2)}")
    print(f"[INFO] Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
