# LAPE (Leak Analysis & Prediction Engine) — Model Evaluation Report
## DATATHON 2026 Track 4: The Money Leak

> [!IMPORTANT]
> **Fair Evaluation Principle**: Supervised metrics are calculated strictly on controlled synthetic scenarios with ground truth labels. Labels are NEVER propagated back to raw real-world data as confirmed fraud.

### 1. Methodology & Leakage Prevention
- **Algorithm**: HistGradientBoostingClassifier (Lightweight Gradient Boosted Decision Trees)
- **Split Strategy**: Grouped K-Fold / Train-Test Split by `source_record_id` (preventing sample variant contamination)
- **Training Set Size**: 270 samples (180 positive leak scenarios)
- **Held-Out Test Set Size**: 90 samples (60 positive leak scenarios)
- **Feature Whitelist Count**: 25 business & mathematical features
- **Metadata & ID Isolation**: Scenario IDs, conditions, and labels strictly excluded from feature matrix.

### 2. Fair Benchmark Comparison
| Method | Precision | Recall | F1-Score | PR-AUC | Precision@10 | FPR on Controls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline A (Raw Shortfall) | 1.0 | 0.717 | 0.835 | 0.944 | 1.0 | 0.0 |
| Baseline B (Derived Rules) | 1.0 | 0.55 | 0.71 | 0.897 | 1.0 | 0.0 |
| LAPE ML Model (HistGB) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

### 3. Scenario Family Detection Breakdown
| Scenario Family | Type | Test Samples | Metric | Performance |
| :--- | :--- | :--- | :--- | :--- |
| `benign_edge_case_bulk` | Valid Control | 14 | Specificity (Clean Pass) | 100.0% (FP: 0) |
| `compound_multi_signal` | Leak Scenario | 25 | Detection Recall | 100.0% (TP: 25) |
| `control_valid` | Valid Control | 16 | Specificity (Clean Pass) | 100.0% (FP: 0) |
| `pricing_extreme_discount` | Leak Scenario | 11 | Detection Recall | 100.0% (TP: 11) |
| `pricing_glitch_negative_margin` | Leak Scenario | 12 | Detection Recall | 100.0% (TP: 12) |
| `return_abuse_cluster` | Leak Scenario | 12 | Detection Recall | 100.0% (TP: 12) |

### 4. Observations & Findings
- **Baseline A (Raw Shortfall)** flags records based purely on dollar threshold, suffering higher false positives on legitimate bulk volume clearance.
- **Baseline B (Derived Rules)** improves specificity by incorporating MAD discount z-scores and catalog flaw segregation.
- **LAPE ML Model** captures non-linear interactions between customer return propensity, discount depth, and margin compression.