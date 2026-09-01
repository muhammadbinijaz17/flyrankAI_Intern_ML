import os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

print("Testing validation audit script...")
df = pd.read_csv("data/raw/content_refresh_anonymized.csv")

# 1. Binary target
df["is_declining_label"] = (df["trend_direction"] == "down").astype(int)

# 2. Features
df["log_impressions_90d"] = np.log1p(df["impressions_90d"].clip(lower=0))
df["log_clicks_90d"] = np.log1p(df["clicks_90d"].clip(lower=0))
df["log_sessions_90d"] = np.log1p(df["sessions_90d"].clip(lower=0))
df["log_pageviews_90d"] = np.log1p(df["pageviews_90d"].clip(lower=0))
df["log_search_volume"] = np.log1p(df["search_volume"].fillna(0).clip(lower=0))

df["has_clicks"] = (df["clicks_90d"] > 0).astype(int)
df["has_sessions"] = (df["sessions_90d"] > 0).astype(int)
df["has_search_volume"] = df["search_volume"].notna().astype(int)
df["has_word_count"] = df["word_count"].notna().astype(int)
df["has_position_data"] = (df["avg_position"] > 0).astype(int)

df["click_through_rate"] = df["ctr"]
df["engagement_rate_safe"] = df["engagement_rate"].fillna(0)
df["scroll_rate_safe"] = df["scroll_rate"].fillna(0)

NUMERIC_FEATURES = [
    "log_impressions_90d",
    "log_clicks_90d",
    "log_sessions_90d",
    "log_pageviews_90d",
    "log_search_volume",
    "days_since_last_update",
    "content_age_days",
    "avg_position",
    "click_through_rate",
    "engagement_rate_safe",
    "scroll_rate_safe",
    "word_count",
    "competition",
    "cpc",
    "days_with_impressions",
    "days_with_sessions",
    "has_clicks",
    "has_sessions",
    "has_search_volume",
    "has_word_count",
    "has_position_data",
]

CATEGORICAL_FEATURES = [
    "content_type",
    "competition_level",
    "main_intent",
    "age_tier",
    "freshness_tier",
    "impression_tier",
    "position_tier",
]

X_num = df[NUMERIC_FEATURES].copy()
for col in ["word_count", "competition", "cpc"]:
    X_num[col] = X_num[col].fillna(0)
X_num = X_num.fillna(0)

X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES].fillna("unknown"), drop_first=True, dtype=float)
X = pd.concat([X_num, X_cat], axis=1)
y = df["is_declining_label"]
groups = df["client_id"]

def evaluate_ranking(y_true, scores, ks=[10, 20, 50, 100, 500]):
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(scores)}).sort_values("score", ascending=False)
    metrics = {}
    for k in ks:
        metrics[f"P@{k}"] = float(frame.head(min(k, len(frame)))["y"].mean())
    metrics["ROC-AUC"] = float(roc_auc_score(y_true, scores))
    metrics["PR-AUC"] = float(average_precision_score(y_true, scores))
    metrics["Base_Rate"] = float(np.mean(y_true))
    return metrics

# Test 1: Random Split (Before)
X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Test 2: Honest Grouped Split (After - Unseen Clients)
np.random.seed(42)
unique_clients = np.sort(df["client_id"].unique())
shuffled_clients = np.random.permutation(unique_clients)
test_client_count = max(1, int(round(len(shuffled_clients) * 0.20)))
test_clients = set(shuffled_clients[:test_client_count])

test_mask = df["client_id"].isin(test_clients)
X_train_grp, X_test_grp = X[~test_mask].copy(), X[test_mask].copy()
y_train_grp, y_test_grp = y[~test_mask].copy(), y[test_mask].copy()

# Fit Random Forest on Random Split
rf_rnd = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=25, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
rf_rnd.fit(X_train_rnd, y_train_rnd)
scores_rnd = rf_rnd.predict_proba(X_test_rnd)[:, 1]
res_rnd = evaluate_ranking(y_test_rnd, scores_rnd)

# Fit Random Forest on Grouped Split
rf_grp = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=25, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
rf_grp.fit(X_train_grp, y_train_grp)
scores_grp = rf_grp.predict_proba(X_test_grp)[:, 1]
res_grp = evaluate_ranking(y_test_grp, scores_grp)

# Fit Logistic Regression on Random Split
lr_rnd = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
lr_rnd.fit(X_train_rnd, y_train_rnd)
lr_scores_rnd = lr_rnd.predict_proba(X_test_rnd)[:, 1]
lr_res_rnd = evaluate_ranking(y_test_rnd, lr_scores_rnd)

# Fit Logistic Regression on Grouped Split
lr_grp = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
lr_grp.fit(X_train_grp, y_train_grp)
lr_scores_grp = lr_grp.predict_proba(X_test_grp)[:, 1]
lr_res_grp = evaluate_ranking(y_test_grp, lr_scores_grp)

print("--- RANDOM SPLIT (BEFORE) ---")
print("RF:", res_rnd)
print("LR:", lr_res_rnd)

print("\n--- GROUPED SPLIT (AFTER) ---")
print("RF:", res_grp)
print("LR:", lr_res_grp)

# Leakage Injection Test
X_leaky = X.copy()
X_leaky["trend_pct"] = df["trend_pct"]
rf_leaky = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_leaky.fit(X_leaky[~test_mask], y[~test_mask])
scores_leaky = rf_leaky.predict_proba(X_leaky[test_mask])[:, 1]
res_leaky = evaluate_ranking(y[test_mask], scores_leaky)
print("\n--- LEAKAGE INJECTION STRESS TEST ---")
print("Leaky Model with trend_pct:", res_leaky)
