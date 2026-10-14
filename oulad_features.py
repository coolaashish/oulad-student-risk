"""Load OULAD tables and build leakage-safe features at a chosen cut-off day.

Only information available up to `cutoff_day` is used, so the model behaves like
a real early-warning system. Students who withdrew before the cut-off are dropped
because their outcome is already known.
"""
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]

CATEGORICAL = [
    "code_module", "gender", "region", "highest_education",
    "imd_band", "age_band", "disability",
]
NUMERIC = [
    "num_of_prev_attempts", "studied_credits",
    "n_assess_submitted", "mean_score", "min_score",
    "total_clicks", "active_days", "clicks_last14",
]


def load_tables(data_dir):
    """Read the five OULAD CSVs we need from data_dir."""
    d = Path(data_dir)
    t = {
        "info": pd.read_csv(d / "studentInfo.csv"),
        "reg": pd.read_csv(d / "studentRegistration.csv"),
        "assess": pd.read_csv(d / "assessments.csv"),
        "sassess": pd.read_csv(d / "studentAssessment.csv"),
        "vle": pd.read_csv(
            d / "studentVle.csv",
            usecols=KEY + ["date", "sum_click"],
        ),
    }
    return t


def build_features(t, cutoff_day=60):
    """Return one row per student-module-presentation with features and target."""
    info = t["info"].copy()
    reg = t["reg"][KEY + ["date_unregistration"]]

    # Drop students whose outcome is already known at the cut-off.
    info = info.merge(reg, on=KEY, how="left")
    still_in = info["date_unregistration"].isna() | (info["date_unregistration"] >= cutoff_day)
    info = info[still_in].drop(columns="date_unregistration")

    # Assessments submitted up to the cut-off.
    a = t["assess"][["id_assessment", "code_module", "code_presentation"]]
    sa = t["sassess"].merge(a, on="id_assessment", how="inner")
    sa = sa[sa["date_submitted"] <= cutoff_day]
    assess_feat = (
        sa.groupby(KEY)
        .agg(
            n_assess_submitted=("id_assessment", "count"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
        )
        .reset_index()
    )

    # VLE (online learning platform) activity up to the cut-off.
    v = t["vle"]
    v = v[v["date"] <= cutoff_day]
    vle_feat = (
        v.groupby(KEY)
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )
    recent = v[v["date"] > cutoff_day - 14]
    recent_feat = (
        recent.groupby(KEY)["sum_click"].sum().rename("clicks_last14").reset_index()
    )

    df = (
        info.merge(assess_feat, on=KEY, how="left")
        .merge(vle_feat, on=KEY, how="left")
        .merge(recent_feat, on=KEY, how="left")
    )
    for col in ["n_assess_submitted", "total_clicks", "active_days", "clicks_last14"]:
        df[col] = df[col].fillna(0)  # no record means no activity
    # mean_score / min_score stay NaN when nothing was submitted; models impute.

    df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    return df


def split_xy(df):
    """Return X, y and the student-id groups used for grouped cross-validation."""
    X = df[CATEGORICAL + NUMERIC]
    y = df["at_risk"]
    groups = df["id_student"]
    return X, y, groups
