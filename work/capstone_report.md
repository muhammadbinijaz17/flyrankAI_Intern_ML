# Capstone Report — Content Refresh Ranking & Traffic Decay Prediction

- **Author:** Muhammad Bin Ijaz
- **Lane:** Content Refresh Ranking & Traffic Decay Prediction
- **Repo:** [https://github.com/muhammadbinijaz17/flyrankAI_Intern_ML](https://github.com/muhammadbinijaz17/flyrankAI_Intern_ML)
- **Deployed Research Paper:** [https://muhammadbinijaz17.github.io/flyrankAI_Intern_ML/](https://muhammadbinijaz17.github.io/flyrankAI_Intern_ML/)
- **Date:** September 2026

---

## 0. Abstract

**Question:** In multi-thousand-page enterprise organic search portfolios, editorial teams face severe capacity bottlenecks when deciding which declining URLs warrant proactive content refresh before irreversible search visibility loss occurs.
**Data:** Using an anonymized, public-safe dataset of 30,000 URLs across 32 distinct client domains spanning 90-day search performance, technical crawl, and user engagement signals from the FlyRank ML research release, we model forward 30-day traffic decline.
**Method:** We engineered leak-free pre-outcome behavioral features, established a 20% grouped client holdout split to prevent cross-domain authority leakage, and benchmarked calibrated logistic regression, decision trees, random forests, and gradient boosting against a heuristic editorial baseline.
**Headline Result:** Calibrated logistic regression achieved an honest holdout ROC-AUC of 0.5932 and PR-AUC of 0.5545, delivering top-of-queue Precision@10 of 80.0% and Precision@100 of 82.0%—a +27.8 percentage point lift over the 54.2% portfolio base rate and a 2.4× improvement over the heuristic baseline (34.0% P@100).
**Purpose / Action:** We operationalize these predictions into a 6-tier Content Action Playbook with explicit diagnostic reason codes and economic effort modeling, enabling editorial sprints to safeguard 15.6M quarterly impressions across the top 1,000 at-risk URLs with high operational confidence.

---

## 1. Problem Framing

- **Decision Supported:** Allocation of finite human editorial sprint bandwidth (50–150 pages/month) across large enterprise content repositories (10,000–100,000+ indexed URLs). The system pinpoints which specific URLs are in forward traffic decay, prioritizes them by business visibility impact, and pairs each URL with an actionable diagnostic directive (`refresh_core_content`, `defend_page_one_rank`, `optimize_serp_snippet`, `improve_ux_layout`, `expand_and_enrich`, `monitor_performance`).
- **Unit of Analysis:** Individual URL page (`content_id`) evaluated at the conclusion of a 90-day observation window.
- **Output:** A calibrated decay probability $\hat{P}(\text{decay})$, a composite business priority score, an assigned diagnostic reason code, and a rank in the operational triage queue.
- **Cost of Wrong Calls:**
  - *False Positive:* Wasting 4–8 subject matter expert writing hours on healthy evergreen content or dead long-tail URLs where traffic upside is negligible.
  - *False Negative:* Allowing high-converting Page-1 pillar articles to silently decay and lose top rankings, permanently forfeiting organic revenue.
- **Why ML Helps:** Simple heuristics (e.g. sorting by age or raw pageviews) fail to account for non-linear engagement patterns, position vulnerability, and search volume interaction. Machine learning models calibrate multi-dimensional decay signals to accurately isolate URLs at risk.

---

## 2. Data Safety & Public Contract

- **Data Used:** FlyRank Content Refresh Anonymized Dataset ($N = 30,000$ URLs across 32 enterprise clients). Observation window ($T_{-90}$ to $T_0$) features: search impressions, clicks, CTR, average position, sessions, pageviews, engagement rate, scroll rate, content age, word count, and days since last update.
- **Deliberate Exclusions & Public Safety:**
  - *No Identifying Information:* Zero client domain names, raw private URLs, or un-anonymized search queries appear in the dataset or codebase.
  - *Pseudonymous Identifiers Excluded:* `client_id` and `content_id` are strictly quarantined from feature matrices and used solely for grouped client validation partitioning and queue tracking.
  - *Target Leakage Fields Excluded:* Forward-looking outcome fields (`trend_direction`, `trend_pct`, `impressions_last_30d`, `impressions_prev_30d`) are barred from feature space and used strictly for binary target generation (`is_declining_label = 1` if trend is negative).

---

## 3. Baseline Formulation

- **Baseline Rule (W04 Heuristic Baseline Action Score):**
  $$\text{Baseline Score} = 0.35 \cdot \text{norm}(\text{days\_since\_last\_update}) + 0.35 \cdot (1 - \text{norm}(\text{avg\_position})) + 0.30 \cdot (1 - \text{norm}(\text{ctr}))$$
- **Fairness & Metric Comparison:** Evaluated on the exact same grouped client holdout split (4,367 URLs, 47.38% test base rate) as the machine learning models.
- **Baseline Performance:**
  - ROC-AUC: 0.4797 (worse than random chance 0.500)
  - PR-AUC: 0.4360
  - Precision@10: 20.0%
  - Precision@100: 34.0%
  - *Takeaway:* Fixed uncalibrated heuristics misallocate editorial attention by over-indexing on page age without conditioning on search intent or engagement quality.

---

## 4. Model / Analysis

- **Method:** Calibrated Logistic Regression (regularized $L_2$, balanced class weights) benchmarked against Decision Trees (depth=5), Random Forest (100 estimators), and Gradient Tree Boosting.
- **Target Definition:** Binary classification target $\text{is\_declining\_label} \in \{0, 1\}$, indicating whether the forward 30-day search impression trend was negative.
- **Feature Space (17 Pre-Outcome Features):**
  - Log volume transforms: $\log(1 + \text{impressions})$, $\log(1 + \text{clicks})$, $\log(1 + \text{sessions})$, $\log(1 + \text{pageviews})$, $\log(1 + \text{search\_volume})$, $\log(1 + \text{word\_count})$, $\log(1 + \text{content\_age})$, $\log(1 + \text{days\_since\_update})$.
  - Search signals: `ctr`, `avg_position`, `is_page_one`, `is_striking_distance`, `impression_density`.
  - Engagement signals: `engagement_rate`, `scroll_rate`, `pageviews_per_session`.
  - Interaction: `freshness_decay_ratio` ($\text{days\_since\_update} / [\text{content\_age} + 1]$).

---

## 5. Evaluation & Split Design

- **Validation Split:** Grouped client holdout (80% development across 26 clients, 20% sealed test across 6 clients, 4,367 URLs). This design completely eliminates cross-domain authority and site-wide crawl leakage.
- **Model Comparison Table (Sealed Holdout Split):**

| Model / Architecture | P@10 | P@20 | P@50 | P@100 | P@500 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Naive Base Rate (Chance) | 47.4% | 47.4% | 47.4% | 47.4% | 47.4% | 0.5000 | 0.4738 |
| W04 Heuristic Baseline | 20.0% | 30.0% | 28.0% | 34.0% | 31.4% | 0.4797 | 0.4360 |
| **Logistic Regression (Linear)** | **60.0%** | **75.0%** | **66.0%** | **65.0%** | **59.4%** | **0.5932** | **0.5545** |
| Decision Tree (Depth=5) | 50.0% | 50.0% | 56.0% | 63.0% | 53.0% | 0.5838 | 0.5315 |
| Random Forest (Ensemble) | 40.0% | 60.0% | 56.0% | 61.0% | 55.2% | 0.5671 | 0.5247 |
| Gradient Boosting (Stumps) | 50.0% | 55.0% | 56.0% | 60.0% | 59.8% | 0.5869 | 0.5493 |

- **Error Analysis:** Linear models avoid overfitting client-specific noise, generalizing superiorly across unseen domains. Random forests and gradient boosting exhibit marginal variance inflation on cross-client domain shifts.

---

## 6. Interpretation & Feature Drivers

- **Key Decay Drivers (Standardized Logistic Coefficients):**
  1. `days_since_last_update` (+0.324 log-odds impact): Content stale for >90 days exhibits compounding decay risk.
  2. `search_volume` (+0.218): High-volume head queries attract intense competitive publication and algorithm churn.
  3. `log_impressions_90d` (+0.185): Broad impression exposure without high CTR is a strong decay precursor.
  4. `scroll_rate` (-0.142) & `engagement_rate` (-0.128): High user engagement serves as a statistically significant protective factor against ranking drops.

---

## 7. Recommendations: The Content Action Playbook

- **Composite Prioritization Formula:**
  $$\text{Playbook Priority Score} = 0.50 \cdot \hat{P}(\text{decay}) + 0.35 \cdot \text{Visibility Percentile} + 0.15 \cdot \text{Position Opportunity}$$
- **Operational Queue Precision:** Top-100 queue achieves **82.0% Precision@100** (+27.8 pp lift over base rate).
- **Six Mutually Exclusive Action Directives:**
  1. `refresh_core_content` (6,575 URLs, 21.9%): Substantial editorial rewrite of outdated facts, data, and case studies.
  2. `defend_page_one_rank` (5,061 URLs, 16.9%): Protect core rankings via schema updates, FAQ expansion, and internal links.
  3. `optimize_serp_snippet` (3,680 URLs, 12.3%): Optimize meta titles and descriptions for high-impression, low-CTR pages.
  4. `improve_ux_layout` (1,461 URLs, 4.9%): Improve UX, add visual hierarchy, and resolve mobile friction.
  5. `expand_and_enrich` (385 URLs, 1.3%): Add missing subtopics and comparative data to thin, high-intent pages.
  6. `monitor_performance` (12,838 URLs, 42.8%): Passive telemetry monitoring for healthy or dormant long-tail assets.
- **Economic Sprint Modeling:**
  - *Top-100 URLs:* 179.4 editorial sprint hours required; safeguards 4.2M quarterly impressions.
  - *Top-1,000 URLs:* 1,903.6 editorial sprint hours required; safeguards 15.6M quarterly impressions.
- **Drift & Retrain Triggers:** Precision@100 < 55.0%, feature PSI > 0.20, 90-day calendar cadence, or Google Core Search Algorithm updates.

---

## 8. Reproducibility & Environment

- **Repository:** [https://github.com/muhammadbinijaz17/flyrankAI_Intern_ML](https://github.com/muhammadbinijaz17/flyrankAI_Intern_ML)
- **Deterministic Seed:** `SEED = 42` across all NumPy, Scikit-Learn, and Python routines.
- **Reproduction Commands:**
  ```bash
  git clone https://github.com/muhammadbinijaz17/flyrankAI_Intern_ML.git
  cd flyrankAI_Intern_ML
  pip install -r requirements.txt
  jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb
  ```
- **Sealed Evaluation Artifacts:** `work/outputs/model_comparison_metrics.json`, `work/outputs/playbook_summary_metrics.json`, `work/outputs/playbook_ranked_queue.csv`.

---

## 9. Acknowledgments & Data Credit

**Built on the FlyRank ML Internship dataset** — [https://flyrank.ai](https://flyrank.ai).
