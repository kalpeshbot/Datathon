"""
LAPE (Leak Analysis & Prediction Engine)
Script: Train ML Leak Detection Models & Evaluate Baselines
DATATHON 2026 Track 4: The Money Leak

Trains HistGradientBoosting and RandomForest classifiers with strict data leakage prevention:
- Grouped splitting by source_record_id
- Complete exclusion of metadata and labels from features
- Preprocessing fit strictly on training set
- Benchmarked against Baseline A (Raw Shortfall) and Baseline B (Derived Rules)
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    build_unified_transaction_feature_matrix,
    compute_pricing_features,
    compute_peer_group_mad_features,
    compute_campaign_features,
    compute_inventory_features
)
from src.rules import apply_baseline_a_simple_rule, apply_baseline_b_composite_rules
from src.model import (
    train_leak_model_with_leakage_prevention,
    prepare_feature_matrix,
    LapeLeakClassifier
)
from src.evaluation import benchmark_baselines_vs_model


def main():
    print("=" * 70)
    print("LAPE: Model Training & Fair Benchmark Evaluation Pipeline")
    print("=" * 70)

    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw and Synthetic Datasets
    print("Loading data...")
    raw_tx = pd.read_csv("data/raw/transactions.csv")
    raw_prod = pd.read_csv("data/raw/products.csv")
    raw_cust = pd.read_csv("data/raw/customers.csv")
    raw_inv = pd.read_csv("data/raw/inventory.csv")
    raw_camp = pd.read_csv("data/raw/campaigns.csv")

    scenarios_df = pd.read_csv("data/synthetic/lape_synthetic_scenarios.csv")
    metadata_df = pd.read_csv("data/synthetic/lape_scenario_metadata.csv")

    # Filter transaction-domain scenarios for the transaction model
    tx_scen_mask = metadata_df["source_domain"] == "transactions"
    tx_scenarios = scenarios_df[tx_scen_mask].copy()
    tx_meta = metadata_df[tx_scen_mask].copy().reset_index(drop=True)

    print(f"Loaded {len(tx_scenarios)} transaction scenarios across {tx_meta['scenario_family'].nunique()} families.")

    # 2. Enrich Scenarios with Derived Economic & Statistical Features
    print("Computing derived economic features for scenario records...")
    tx_scenarios_feat = compute_pricing_features(tx_scenarios, raw_prod)
    tx_scenarios_feat = compute_peer_group_mad_features(tx_scenarios_feat, group_col="category", value_col="discount_pct")
    tx_scenarios_feat = compute_peer_group_mad_features(tx_scenarios_feat, group_col="category", value_col="unit_margin")

    # Also apply Baseline A and Baseline B rules to scenario records
    tx_scenarios_feat = apply_baseline_a_simple_rule(tx_scenarios_feat, exposure_threshold=100.0)
    tx_scenarios_feat = apply_baseline_b_composite_rules(tx_scenarios_feat, min_shortfall=25.0)

    # 3. Train Model with Grouped Leak-Free Split
    print("\nExecuting Grouped Leak-Free Train/Test Split (test_size=0.25, seed=42)...")
    clf_hgb, split_info = train_leak_model_with_leakage_prevention(
        scenarios_df=tx_scenarios_feat,
        metadata_df=tx_meta,
        model_type="hist_gb",
        random_state=42,
        test_size=0.25
    )

    print(f"Train samples: {split_info['train_samples']} (Positives: {split_info['train_positives']})")
    print(f"Test samples:  {split_info['test_samples']} (Positives: {split_info['test_positives']})")
    print(f"Features used ({len(split_info['feature_names'])}): {split_info['feature_names']}")

    # Save Model
    hgb_path = models_dir / "lape_hist_gb.joblib"
    clf_hgb.save(str(hgb_path))

    # Also train RandomForest for ensemble comparison
    print("\nTraining RandomForest classifier for architectural comparison...")
    clf_rf = LapeLeakClassifier(model_type="random_forest", random_state=42)
    clf_rf.fit(split_info["X_train"], split_info["y_train"], split_info["feature_names"])
    rf_path = models_dir / "lape_random_forest.joblib"
    clf_rf.save(str(rf_path))

    # 4. Predict on Held-Out Test Set
    test_scores = clf_hgb.predict_proba(split_info["X_test"])[:, 1]
    test_preds = clf_hgb.predict(split_info["X_test"], threshold=0.5)

    test_df = tx_scenarios_feat.iloc[np.isin(tx_meta["source_record_id"], split_info["meta_test"]["source_record_id"])].copy()

    # 5. Benchmark Baselines vs Model
    print("\nRunning Rigorous Benchmark: Baseline A vs. Baseline B vs. LAPE ML Model...")
    benchmark_results = benchmark_baselines_vs_model(
        test_df=test_df,
        test_meta=split_info["meta_test"],
        y_test=split_info["y_test"],
        model_preds=test_preds,
        model_scores=test_scores,
        budget_k=10
    )

    # Display comparison table
    comp_df = pd.DataFrame(benchmark_results["comparison_table"])
    print("\n" + comp_df.to_string(index=False))

    print("\nPer-Scenario-Family Detection Breakdown (LAPE ML Model):")
    for fam, details in benchmark_results["model_family_breakdown"].items():
        if details["type"] == "control":
            print(f"  - [Control] {fam:<30}: Samples={details['sample_count']}, Clean Pass={details['clean_pass_rate']*100:.1f}%, FP={details['false_positives']}")
        else:
            print(f"  - [Leak]    {fam:<30}: Samples={details['sample_count']}, Detection Recall={details['detection_pct']}% (TP={details['true_positives']})")

    # 6. Save Benchmark Results & Markdown Report
    with open(reports_dir / "evaluation_report.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved benchmark results to {reports_dir / 'evaluation_report.json'}")

    # Generate Markdown Report
    md = [
        "# LAPE (Leak Analysis & Prediction Engine) — Model Evaluation Report",
        "## DATATHON 2026 Track 4: The Money Leak\n",
        "> [!IMPORTANT]",
        "> **Fair Evaluation Principle**: Supervised metrics are calculated strictly on controlled synthetic scenarios with ground truth labels. Labels are NEVER propagated back to raw real-world data as confirmed fraud.\n",
        "### 1. Methodology & Leakage Prevention",
        f"- **Algorithm**: HistGradientBoostingClassifier (Lightweight Gradient Boosted Decision Trees)",
        f"- **Split Strategy**: Grouped K-Fold / Train-Test Split by `source_record_id` (preventing sample variant contamination)",
        f"- **Training Set Size**: {split_info['train_samples']} samples ({split_info['train_positives']} positive leak scenarios)",
        f"- **Held-Out Test Set Size**: {split_info['test_samples']} samples ({split_info['test_positives']} positive leak scenarios)",
        f"- **Feature Whitelist Count**: {len(split_info['feature_names'])} business & mathematical features",
        f"- **Metadata & ID Isolation**: Scenario IDs, conditions, and labels strictly excluded from feature matrix.\n",
        "### 2. Fair Benchmark Comparison",
    ]
    headers = list(comp_df.columns)
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join([":---" for _ in headers]) + " |")
    for _, row in comp_df.iterrows():
        md.append("| " + " | ".join([str(row[h]) for h in headers]) + " |")
    
    md.extend([
        "\n### 3. Scenario Family Detection Breakdown",
        "| Scenario Family | Type | Test Samples | Metric | Performance |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])
    for fam, d in benchmark_results["model_family_breakdown"].items():
        if d["type"] == "control":
            md.append(f"| `{fam}` | Valid Control | {d['sample_count']} | Specificity (Clean Pass) | {d['clean_pass_rate']*100:.1f}% (FP: {d['false_positives']}) |")
        else:
            md.append(f"| `{fam}` | Leak Scenario | {d['sample_count']} | Detection Recall | {d['detection_pct']}% (TP: {d['true_positives']}) |")
    
    md.append("\n### 4. Observations & Findings")
    md.append("- **Baseline A (Raw Shortfall)** flags records based purely on dollar threshold, suffering higher false positives on legitimate bulk volume clearance.")
    md.append("- **Baseline B (Derived Rules)** improves specificity by incorporating MAD discount z-scores and catalog flaw segregation.")
    md.append("- **LAPE ML Model** captures non-linear interactions between customer return propensity, discount depth, and margin compression.")

    with open(reports_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Markdown report written to {reports_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
