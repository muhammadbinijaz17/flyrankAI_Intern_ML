import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load data
df = pd.read_csv("data/raw/content_refresh_anonymized.csv")
df["is_declining_label"] = (df["trend_direction"] == "down").astype(int)

# Recompute baseline_action_score identically to w04
def percentile_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return values.rank(method="average", pct=True).fillna(0)

def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    mi, ma = values.min(), values.max()
    if not np.isfinite(mi) or not np.isfinite(ma) or mi == ma:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mi) / (ma - mi)

df["visibility_score"] = percentile_rank(np.log1p(df["impressions_90d"]))
df["freshness_risk_score"] = percentile_rank(df["days_since_last_update"])
df["position_opportunity_score"] = (
    (1.0 - normalize(df["avg_position"].clip(lower=1, upper=50)))
    * df["visibility_score"]
    * (df["avg_position"] > 0).astype(int)
)
df["depth_gap_score"] = (1.0 - percentile_rank(df["word_count"].fillna(df["word_count"].median()))) * df["visibility_score"]
df["baseline_action_score"] = (
    0.40 * df["visibility_score"]
    + 0.30 * df["freshness_risk_score"]
    + 0.25 * df["position_opportunity_score"]
    + 0.05 * df["depth_gap_score"]
).clip(0, 1)

# Feature engineering
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

num_cols = [
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

cat_cols = [
    "content_type",
    "competition_level",
    "main_intent",
    "age_tier",
    "freshness_tier",
    "impression_tier",
    "position_tier",
]

X_num = df[num_cols].copy()
for col in ["word_count", "competition", "cpc"]:
    X_num[col] = X_num[col].fillna(0)
X_num = X_num.fillna(0)

X_cat = pd.get_dummies(df[cat_cols].fillna("unknown"), drop_first=True, dtype=float)

X = pd.concat([X_num, X_cat], axis=1)
y = df["is_declining_label"]

# Grouped Split (Holdout 20% clients for honest unseen client test)
np.random.seed(42)
unique_clients = np.sort(df["client_id"].unique())
shuffled_clients = np.random.permutation(unique_clients)
test_client_count = max(1, int(round(len(shuffled_clients) * 0.2)))
test_clients = set(shuffled_clients[:test_client_count])

test_mask = df["client_id"].isin(test_clients)
train_mask = ~test_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
df_test = df[test_mask].copy()

# Logistic Regression
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_scores = lr_pipe.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=25,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_scores = rf.predict_proba(X_test)[:, 1]

# Permutation importance on test set for Random Forest
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc")
perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm_importance.importances_mean,
    "importance_std": perm_importance.importances_std,
}).sort_values("importance_mean", ascending=False)

print("\n--- TOP 10 PERMUTATION IMPORTANCE (TEST SET ROC-AUC DROP) ---")
print(perm_df.head(10).to_string(index=False))

# Coefficients for Logistic Regression
lr_model = lr_pipe.named_steps["model"]
lr_coefs = pd.DataFrame({
    "feature": X.columns,
    "coefficient": lr_model.coef_[0],
    "abs_coef": np.abs(lr_model.coef_[0])
}).sort_values("abs_coef", ascending=False)

print("\n--- TOP 10 LOGISTIC REGRESSION COEFFICIENTS ---")
print(lr_coefs.head(10).to_string(index=False))

# Error Analysis
df_test["rf_score"] = rf_scores
df_test["lr_score"] = lr_scores
df_test["rf_pred"] = (rf_scores >= 0.5).astype(int)

# False Positives in Top Ranked items
top50_rf = df_test.sort_values("rf_score", ascending=False).head(50)
fp_cases = top50_rf[top50_rf["is_declining_label"] == 0]
print(f"\nFalse Positives in Top 50 (Predicted High Risk, but Actually Stable): {len(fp_cases)}")
print(fp_cases[["content_id", "client_id", "rf_score", "impressions_90d", "avg_position", "days_since_last_update", "word_count"]].head(5).to_string(index=False))

# False Negatives (Predicted Low Risk, but Actually Declining)
bottom50_rf = df_test.sort_values("rf_score", ascending=True).head(50)
fn_cases = bottom50_rf[bottom50_rf["is_declining_label"] == 1]
print(f"\nFalse Negatives in Bottom 50 (Predicted Low Risk, but Actually Declining): {len(fn_cases)}")
print(fn_cases[["content_id", "client_id", "rf_score", "impressions_90d", "avg_position", "days_since_last_update", "word_count"]].head(5).to_string(index=False))
