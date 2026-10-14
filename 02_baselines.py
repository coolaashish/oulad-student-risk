"""Baseline models for early student-risk prediction on OULAD.

Usage:
    python 02_baselines.py --data-dir data/oulad --cutoff 60
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from oulad_features import CATEGORICAL, NUMERIC, build_features, load_tables, split_xy


def make_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def make_models():
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=42,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/oulad")
    ap.add_argument("--cutoff", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="baseline_results.csv")
    args = ap.parse_args()

    tables = load_tables(args.data_dir)
    df = build_features(tables, cutoff_day=args.cutoff)
    X, y, groups = split_xy(df)
    print(f"Rows: {len(df)} | students: {groups.nunique()} | at-risk rate: {y.mean():.3f} | cutoff day: {args.cutoff}")

    cv = GroupKFold(n_splits=args.folds)
    records = []
    for name, model in make_models().items():
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
            t0 = time.perf_counter()
            pipe.fit(X.iloc[tr], y.iloc[tr])
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            proba = pipe.predict_proba(X.iloc[te])[:, 1]
            pred_s = time.perf_counter() - t0
            pred = (proba >= 0.5).astype(int)

            records.append({
                "model": name, "fold": fold, "cutoff": args.cutoff,
                "accuracy": accuracy_score(y.iloc[te], pred),
                "f1": f1_score(y.iloc[te], pred, zero_division=0),
                "roc_auc": roc_auc_score(y.iloc[te], proba) if y.iloc[te].nunique() > 1 else np.nan,
                "fit_seconds": fit_s,
                "predict_ms_per_student": 1000 * pred_s / len(te),
            })

    res = pd.DataFrame(records)
    res.to_csv(args.out, index=False)
    summary = res.groupby("model")[["accuracy", "f1", "roc_auc", "fit_seconds", "predict_ms_per_student"]].agg(["mean", "std"]).round(4)
    print(summary.to_string())
    print(f"\nSaved per-fold results to {args.out}")


if __name__ == "__main__":
    main()
