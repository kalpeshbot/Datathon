# LAPE Engineering Architecture

## 1. System Vision & Constraints

**LAPE (Leak Analysis & Prediction Engine)** is engineered for **DATATHON 2026 Track 4: The Money Leak**. Its primary mandate is to identify, quantify, explain, and prioritize suspicious financial and operational patterns across business data.

### Three Non-Negotiable Output Dimensions
1. **Potential Financial Exposure**: A transparent, reproducible dollar calculation derived directly from verified source fields.
2. **Evidence Reliability**: Categorized into `HIGH`, `MEDIUM`, or `LOW` based on schema constraints, relational joins, and dataset characteristics.
3. **Action Urgency**: Prioritized into `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` according to documented business rules rather than opaque probability scores.

```mermaid
graph TD
    subgraph Data Layer
        C[Customers 3K] --> TX_F[Feature Engineering Pipeline]
        P[Products 250] --> TX_F
        T[Transactions 18K] --> TX_F
        I[Inventory 5K] --> INV_F[Inventory Feature Engine]
        CP[Campaigns 1.2K] --> CAMP_F[Campaign Funnel Engine]
    end

    subgraph Analytical Layer
        TX_F --> BA[Baseline A: Raw Threshold]
        TX_F --> BB[Baseline B: Composite Heuristics]
        TX_F --> ML[HistGradientBoosting Classifier]
        INV_F --> TR[Evidence-Linked Triage Engine]
        CAMP_F --> TR
        BA --> TR
        BB --> TR
        ML --> TR
    end

    subgraph Presentation Layer
        TR --> UQ[Unified Alert Queue]
        UQ --> UI1[01 Overview Executive Dashboard]
        UQ --> UI2[02 Data Quality Live Audit]
        UQ --> UI3[03 Evidence-Linked Alert Queue]
        UQ --> UI4[04 Deep Investigation Console]
        ML --> UI5[05 What-If Scenario Lab]
        UQ --> UI6[06 Model Benchmark vs Rules]
        UQ --> UI7[07 Executive PDF & CSV Reports]
    end
```

## 2. Component Directory Architecture

- **`src/`**: Pure core libraries with zero Streamlit dependencies.
  - `validation.py`: Schema validation, referential integrity verification, funnel physics.
  - `features.py`: Pricing calculations, peer-group MAD z-scores, temporal and customer aggregations.
  - `rules.py`: Transparent baseline heuristics (Baseline A and Baseline B).
  - `scenario_generator.py`: Controlled synthetic scenario generation engine.
  - `model.py`: Model wrappers with target-leakage safeguards and grouped train/test splitting.
  - `evaluation.py`: Precision, recall, PR-AUC, Precision@K, and per-family performance metrics.
  - `triage.py`: Three-dimensional alert generation and calculation provenance.
  - `currency.py`: Indian numbering system formatting (`₹ Lakhs`, `₹ Crores`).
  - `pdf_report.py`: Enterprise PDF generator with ReportLab.

- **`scripts/`**: Automation pipelines for batch operations.
  - `audit_data.py`: CLI data audit and sanity verification.
  - `generate_scenarios.py`: Batch scenario generator producing metadata and synthetic tables.
  - `train_model.py`: End-to-end model training, grouped evaluation, and artifact serialization.
  - `evaluate_model.py`: Model benchmarking script against baselines.

- **`tests/`**: Comprehensive test suite covering equations, scenario physics, triage, and data integrity.
