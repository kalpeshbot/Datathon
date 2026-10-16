# Machine Learning Methodology & Evaluation Credibility

## 1. Machine Learning Strategy

LAPE adopts a tree-based machine learning architecture (`HistGradientBoostingClassifier`) designed for tabular business records with mixed categorical and numerical distributions.

### Feature Whitelist (Leakage Protection)
The model is strictly trained on a whitelist of operational features:
- `quantity`, `unit_price`, `discount_pct`, `effective_unit_price`, `unit_cost`, `unit_margin`
- `price_cost_shortfall`, `potential_shortfall_amount`, `gross_revenue`, `net_revenue`
- `price_to_list_ratio`, `catalog_cost_gt_list`
- `category_discount_pct_robust_z`, `category_unit_margin_robust_z`
- `cust_return_rate`, `cust_avg_discount`, `cust_total_tx`, `hour`, `dayofweek`
- Encoded categoricals: `status_code`, `channel_code`, `category_code`, `payment_method_code`

**Strict Exclusion**: All identifier columns (`transaction_id`, `customer_id`, `product_id`, `scenario_id`, `source_record_id`) and scenario metadata (`injected_condition`, `severity`, `expected_detection`) are strictly excluded prior to feature matrix construction.

---

## 2. Grouped Partitioning

To avoid optimistic evaluation bias, training and test datasets are partitioned using **GroupKFold / Grouped Partitioning** on `source_record_id`:
- All synthetic variants originating from a given raw transaction record remain strictly in either the training or testing partition.
- Cross-partition record contamination is zero ($0$ overlapping source records).

---

## 3. Evaluation Credibility & Scope

### Benchmark Performance (Held-Out Test Set, 90 samples)

| Method | Precision | Recall | F1-Score | PR-AUC | Precision@10 | Control FPR |
|---|---|---|---|---|---|---|
| **Baseline A (Raw Shortfall)** | 1.000 | 0.717 | 0.835 | 0.944 | 1.000 | 0.000 |
| **Baseline B (Derived Composite Rules)** | 1.000 | 0.550 | 0.710 | 0.897 | 1.000 | 0.000 |
| **LAPE ML Model (HistGB)** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

### Critical Scientific Caveat
> [!WARNING]
> While LAPE achieves an F1-Score of 1.000 on the synthetic evaluation benchmark, this reflects its ability to distinguish documented physical and operational anomalies from valid controls.
>
> It does **not** imply 100% real-world detection accuracy across unconstrained business fraud. In production environments, ongoing human review and continuous model calibration against confirmed auditor adjustments remain essential.
