# LAPE (Leak Analysis & Prediction Engine)
### Evidence-Linked Financial Triage & Financial Leak Physics Platform
**DATATHON 2026 — Track 4: The Money Leak**

---

## 1. Project Identity & Overview

**LAPE (Leak Analysis & Prediction Engine)** combines business-based mathematical equations, scientifically constrained synthetic data generation, machine learning, and evidence-linked financial analysis to identify and prioritize potential financial and operational leaks.

> [!IMPORTANT]
> **Core Product Promise & Governance**:
> "Turn suspicious business numbers into evidence-linked financial investigations, with transparent calculations, explicit uncertainty, and measurable detection performance."
> 
> LAPE never fabricates results, claims unverified real-world accuracy, or presents potential financial exposure as confirmed accounting fraud or realized loss.

### Three Distinct Output Dimensions
Throughout the backend and UI, LAPE strictly segregates three critical dimensions:
1. **Potential Financial Exposure**: A transparent, reproducible dollar calculation derived directly from verified source fields.
2. **Evidence Reliability**: Classified into `HIGH`, `MEDIUM`, or `LOW` based on schema constraints, relational joins, and dataset characteristics.
3. **Action Urgency**: Prioritized into `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` by a documented operational policy rather than an opaque model confidence score.

---

## 2. Quickstart & One-Click Launch

### Option A: One-Click Startup (Recommended for Windows)
Double-click [`run_lape.bat`](file:///c:/Users/kalpesh/OneDrive/Desktop/Datathon/run_lape.bat) or run from terminal:
```cmd
run_lape.bat
```
This launcher automatically verifies Python 3.12, confirms installed packages (`pandas`, `numpy`, `scikit-learn`, `streamlit`), and boots the dashboard at `http://localhost:8501`.

### Option B: Manual Terminal Execution
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit Application (Localhost Port 8501)
python -m streamlit run app.py --server.port 8501
```

### Local Evaluation Sandbox Credentials
For local reviewer and evaluator convenience:
- **Default Username:** `demo@lape.local` (or `admin`)
- **Default Password:** `lape2026`
- **Environment Override:** Set `LAPE_DEMO_USER` and `LAPE_DEMO_PASS` in your environment to override.
- *Includes a local arithmetic challenge (CAPTCHA) to simulate authenticated access within an offline evaluation sandbox.*

### Option C: Running the Full Reproducible Pipeline from Scratch
```powershell
# 1. Audit raw datasets & confirm baseline counts
& "C:\Users\kalpesh\AppData\Local\Programs\Python\Python312\python.exe" scripts/audit_data.py

# 2. Generate controlled synthetic scenarios (random seed 42)
& "C:\Users\kalpesh\AppData\Local\Programs\Python\Python312\python.exe" scripts/generate_scenarios.py

# 3. Train ML models & benchmark against baselines with grouped split
& "C:\Users\kalpesh\AppData\Local\Programs\Python\Python312\python.exe" scripts/train_model.py

# 4. Run automated test suite (15 passing tests)
& "C:\Users\kalpesh\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests -v
```

---

## 3. Data Sources & Verified Baseline Audit

LAPE inspects five supplied CSV datasets in `DATASET/` (replicated to `data/raw/` while preserving the originals):

| Dataset | Row Count | Primary Key | Foreign Key Target | Missing Values | Duplicate Rows |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customers.csv` | 3,000 | `customer_id` | — | 0 | 0 |
| `products.csv` | 250 | `product_id` | — | 0 | 0 |
| `transactions.csv` | 18,000 | `transaction_id` | `customer_id`, `product_id` | 0 | 0 |
| `inventory.csv` | 5,000 | `inventory_record_id` | `product_id` | 0 | 0 |
| `campaigns.csv` | 1,200 | `campaign_id` | — | 0 | 0 |
| **Total Inspected** | **27,450** | — | **0 Orphan Keys** | **0** | **0** |

### Verified Audit Findings
- **Transaction Status Distribution**:
  - `Completed`: 13,962 (Fulfilled sales; revenue realized)
  - `Cancelled`: 1,689 (Averted exposure)
  - `Returned`: 1,431 (Customer refunds; inventory restocking risk)
  - `Pending`: 918 (In-flight unfulfilled orders)
- **Mathematical Margin Discrepancy**: 6,829 of 18,000 transactions have effective unit price below unit cost (5,315 Completed sales; total calculated shortfall $1,214,115.50).
- **Catalog Cost Inversion**: 76 of 250 products (30.4%) have `unit_cost > list_price`. As confirmed in dataset documentation, costs and prices were generated independently in this educational dataset.
- **Marketing Funnel Inversions**: 171 of 1,200 campaign records violate physical funnel constraints (84 have `clicks > impressions`, 88 have `conversions > clicks`).
- **Inventory Discrepancy**: 1,253 records disagree with the physical threshold rule (`stock_on_hand <= reorder_threshold` vs `stockout_flag`).

---

## 4. Mathematical Equations & Feature Engineering

All financial equations are explicit, verified by unit tests, and avoid double counting:

### A. Pricing, Margins, and Shortfalls
1. **Effective Unit Price**:
   $$\text{effective\_unit\_price} = \text{unit\_price} \times (1 - \text{discount\_pct})$$
   *(Verified: `discount_pct` is stored as a decimal fraction, e.g., 0.20 = 20%).*
2. **Unit Margin**:
   $$\text{unit\_margin} = \text{effective\_unit\_price} - \text{unit\_cost}$$
3. **Price-Cost Shortfall (Per Unit)**:
   $$\text{price\_cost\_shortfall} = \max(0, \text{unit\_cost} - \text{effective\_unit\_price})$$
4. **Potential Shortfall Amount (Line Item Total)**:
   $$\text{potential\_shortfall\_amount} = \text{quantity} \times \text{price\_cost\_shortfall}$$
5. **Status-Segregated Realization**:
   - `Completed`: Realized shortfall (immediate accounting deficit)
   - `Pending`: In-flight operational risk (prior to shipment intervention)
   - `Returned`: Return cost at risk (refund/restocking deficit)
   - `Cancelled`: Averted exposure ($0 realized loss)

### B. Statistical Peer-Group Deviations (Median & MAD)
To avoid outlier contamination from mean and standard deviation, LAPE calculates robust deviations per product category:
$$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$
$$\text{Robust } z = \frac{x_i - \text{median}(x)}{\max(1.4826 \times \text{MAD}, 10^{-4})}$$
Safeguards handle zero MAD and small peer groups safely without division by zero.

### C. Marketing Funnel Physics
$$\text{Compromised Spend Exposure} = \begin{cases} \text{spend} & \text{if } \text{clicks} > \text{impressions} \lor \text{conversions} > \text{clicks} \\ 0 & \text{otherwise} \end{cases}$$

---

## 5. Financial Leak Physics Scenario Generator

To test detection capability without fabricating labels on raw data, LAPE includes a reproducible scenario generator (`random_seed=42`):

- **600 Total Generated Scenarios** across 10 families:
  1. `control_valid` (60): Unmodified healthy completed sales (Expected Detection = 0)
  2. `pricing_extreme_discount` (60): Rogue unapproved discounts 50%–85% (Expected Detection = 1)
  3. `pricing_glitch_negative_margin` (60): Catalog checkout price tampered below cost (Expected Detection = 1)
  4. `campaign_funnel_inversion` (60): Physical funnel inversions (Expected Detection = 1)
  5. `campaign_budget_burn` (60): High spend with zero clicks/conversions (Expected Detection = 1)
  6. `inventory_phantom_stockout` (60): Flagged 1 with 5x surplus stock (Expected Detection = 1)
  7. `inventory_unflagged_depletion` (60): Zero stock unflagged with 15–30 day supplier delay (Expected Detection = 1)
  8. `return_abuse_cluster` (60): High volume repeated returns (Expected Detection = 1)
  9. `compound_multi_signal` (60): Bulk volume + deep discount + negative margin (Expected Detection = 1)
  10. `benign_edge_case_bulk` (60): Large volume promotional purchase with legitimate positive margin (Expected Detection = 0)

### Leakage Prevention Guarantees
- **Strict Grouped Partitioning**: All scenario variants originating from the same `source_record_id` are grouped into one split. No source record crosses train and test splits.
- **Target Feature Isolation**: `scenario_id`, `injected_condition`, `severity`, and `expected_detection` are strictly excluded from model feature matrices.

---

## 6. Fair Benchmark Evaluation

Models were evaluated on a held-out test split (90 samples: 60 leak injections, 30 valid controls) grouped strictly by source record:

| Method | Precision | Recall | F1-Score | PR-AUC | Precision@10 | FPR on Controls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline A** (Raw Shortfall Ranking) | 1.000 | 0.717 | 0.835 | 0.944 | 1.000 | 0.000 |
| **Baseline B** (Derived Composite Rules) | 1.000 | 0.550 | 0.710 | 0.897 | 1.000 | 0.000 |
| **LAPE ML Model** (HistGradientBoosting) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

### Per-Scenario-Family Breakdown (Held-Out Test Set)
- `benign_edge_case_bulk`: 14 samples — Clean Pass Specificity: **100.0%** (0 False Positives)
- `control_valid`: 16 samples — Clean Pass Specificity: **100.0%** (0 False Positives)
- `compound_multi_signal`: 25 samples — Detection Recall: **100.0%**
- `pricing_extreme_discount`: 11 samples — Detection Recall: **100.0%**
- `pricing_glitch_negative_margin`: 12 samples — Detection Recall: **100.0%**
- `return_abuse_cluster`: 12 samples — Detection Recall: **100.0%**

> [!WARNING]
> **Evaluation Transparency & Non-Generalization**:
> High performance on synthetic test sets reflects separation of documented physical anomalies from valid controls. It does **not** prove 100% detection of real-world corporate fraud. Real-world applications require ongoing human review and model calibration against audited accounting adjustments.

---

## 7. Dashboard Architecture & Interactive Walkthrough

The Streamlit dashboard (`app.py`) provides a professional, dual-theme forensic interface:

- **Theme & Accessibility**: Toggle between **Dark Industrial** and **Clean Light** themes (`ui/themes.py`) with responsive cards and tabular monospace number alignment.
- **Global Multi-Language Localization**: Full UI localization across **8 languages** (`ui/translations.py`): English, Hindi (हिन्दी), Spanish (Español), French (Français), German (Deutsch), Japanese (日本語), Chinese (中文), and Russian (Русский).
- **Dual Currency Support**: Instant toggle between Indian Rupee (₹ INR) and US Dollar ($ USD) with proper comma separators and decimal precision (`src/currency.py`).

### Five Dedicated Forensic Modules:
1. **📊 Data Quality & Pipeline Audit**: Recomputes all dataset statistics, schema validations, foreign key orphan checks, and baseline inconsistency summaries live.
2. **🚨 Evidence-Linked Alert Queue**: Multi-filter alert stream (Domain, Urgency, Reliability, Search). Selecting an alert displays:
   - Dedicated cards for Potential Financial Exposure, Evidence Reliability, and Action Urgency.
   - Step-by-step mathematical calculation formula.
   - Recommended operational verification actions (e.g., hold pending shipment, audit vendor contract).
3. **🔬 Financial Leak Physics Lab**: Interactive simulator allowing managers to manipulate quantity, cost, discount, and status in real-time, observing live margin calculations and ML leak probability predictions.
4. **⚖️ Fair Benchmark (Baselines vs Model)**: Direct side-by-side comparison tables, precision-recall metrics, and per-family performance breakdowns.
5. **📋 Methodology, Provenance & Export**: Mathematical documentation, synthetic scenario manifest, and one-click CSV export of prioritized alert queues.

---

## 8. Test Suite Summary

Run all tests via pytest:
```powershell
& "C:\Users\kalpesh\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests -v
```
**15 of 15 tests passed** (1.08s):
- `test_hand_calculated_positive_margin`: Verifies $0 shortfall on positive margins.
- `test_hand_calculated_negative_margin_shortfall`: Verifies exact line shortfall on negative margins.
- `test_discount_fraction_representation`: Verifies discount fraction scaling (0.15 vs 15).
- `test_transaction_status_segregation`: Verifies Completed vs Pending vs Returned vs Cancelled segregation.
- `test_zero_and_negative_edge_cases`: Verifies zero quantity, zero price, and 100% discount.
- `test_scenario_reproducibility`: Verifies seed reproducibility across multiple generator runs.
- `test_target_leakage_prevention`: Verifies exclusion of scenario metadata from features.
- `test_grouped_split_isolation`: Verifies 0 source record overlap between train and test sets.
- `test_triage_three_distinct_dimensions`: Verifies exposure, reliability, and urgency separation.
- `test_no_double_counting_in_unified_queue`: Verifies unique alert IDs and exact exposure summation.
- `test_hand_calculated_metric_evaluations`: Verifies precision, recall, f1, precision@k, and control FPR.
- `test_raw_row_counts_and_uniqueness`: Verifies 27,450 row counts and 0 duplicates across all 5 CSVs.
- `test_referential_integrity_zero_orphans`: Verifies 0 orphan foreign keys.
- `test_campaign_funnel_physics_detection`: Verifies detection of 171 campaign funnel inversions.
- `test_inventory_flag_discrepancy_detection`: Verifies detection of 1,253 inventory flag discrepancies.

---

## 9. Known Limitations & Operational Considerations

1. **Synthetic Dataset Margin Inversion**: In this educational dataset, 76 products have list price below unit cost. In production, this must be verified against actual supplier purchase agreements before treating as operational fraud.
2. **Campaign-Transaction Linkage**: Campaigns lack transaction-level tracking keys; marketing spend exposure is therefore evaluated at the campaign funnel level rather than attributed to individual orders.
3. **Lack of Audited Financial Loss Labels**: Supervised models are trained and tested on controlled synthetic scenarios. In enterprise deployment, periodic recalibration on confirmed auditor adjustments is necessary.

---

## 10. Configuration & Environment Variables

LAPE is self-contained and operates without required external services. Optional environment variables include:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LAPE_DEMO_USER` | `demo@lape.local` | Username for evaluator login sandbox |
| `LAPE_DEMO_PASS` | `lape2026` | Password for evaluator login sandbox |
| `STREAMLIT_SERVER_PORT` | `8501` | Listening port for the Streamlit web dashboard |
| `STREAMLIT_SERVER_HEADLESS`| `true` | Runs server headlessly without auto-opening browser |

---

## 11. Privacy, Security & Data Handling

- **100% Offline Local Processing**: All mathematical audits, scenario synthesis, model training, inference, and visualization run strictly on the local host.
- **Zero Cloud Data Exfiltration**: No customer data, transactions, inventory counts, or financial values are ever transmitted to any third-party API or cloud service.
- **Local Reviewer Sandbox**: Includes an offline challenge mechanism (arithmetic CAPTCHA) to demonstrate access control patterns suitable for corporate financial intelligence systems.
- **Zero Secret Storage**: The repository contains no API keys, credentials, or private access tokens.

---

## 12. Future Roadmap & Proposed Gemini AI Explanation Layer

> [!NOTE]
> **Implementation Clarification**:
> LAPE currently executes 100% on deterministic mathematical rules, physical funnel constraints, and local scikit-learn models (`HistGradientBoostingClassifier`, `RandomForestClassifier`). **No external LLM or AI API is currently integrated or required for execution.**

### Proposed Future Enhancement:
An optional, enterprise-grade Gemini API explanation layer could assist non-technical operational investigators by translating complex multi-dimensional evidence into concise executive briefing notes.

**Mandatory Grounding Principles for Future AI Integration**:
1. **Strict Evidence Grounding**: The AI model must only receive and summarize verifiable evidence generated by LAPE's existing mathematical engines, rule outputs, and trained model probabilities.
2. **No Hallucinated Predictions**: The language model must never invent new financial shortfall figures, fabricate audit findings, or alter assigned priority scores.
3. **Audit Trail Retention**: Every generated text narrative must link back to the exact record IDs and mathematical formulas that produced the underlying evidence.
