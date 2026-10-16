"""
LAPE (Leak Analysis & Prediction Engine)
Script: Standalone Model Evaluation & Benchmarking
DATATHON 2026 Track 4: The Money Leak

Loads trained models from models/ and evaluates them on held-out test scenarios,
comparing against Baseline A and Baseline B.
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import compute_pricing_features, compute_peer_group_mad_features
from src.rules import apply_baseline_a_simple_rule, apply_baseline_b_composite_rules
from src.model import LapeLeakClassifier, prepare_feature_matrix
from src.evaluation import benchmark_baselines_vs_model


def main():
    print("=" * 70)
    print("LAPE: Standalone Model Evaluation & Baseline Comparison")
    print("=" * 70)

    # 1. Check model existence
    model_path = Path("models/lape_hist_gb.joblib")
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found. Please run scripts/train_model.py first.")
        sys.exit(1)

    # 2. Load synthetic scenarios and metadata
    scenarios_path = Path("data/synthetic/lape_synthetic_scenarios.csv")
    meta_path = Path("data/synthetic/lape_scenario_metadata.csv")
    prod_path = Path("data/raw/products.csv")

    if not scenarios_path.exists() or not meta_path.exists():
        print("Error: Synthetic scenario datasets missing. Run scripts/generate_scenarios.py first.")
        sys.exit(1)

    print("Loading test datasets...")
    scenarios_df = pd.read_csv(scenarios_path)
    metadata_df = pd.read_csv(meta_path)
    raw_prod = pd.read_csv(prod_path)

    # Filter transaction scenarios
    mask = metadata_df["source_domain"] == "transactions"
    tx_scen = scenarios_df[mask].copy()
    tx_meta = metadata_df[mask].copy().reset_index(drop=True)

    print("Computing derived features and rule baselines...")
    tx_feat = compute_pricing_features(tx_scen, raw_prod)
    tx_feat = compute_peer_group_mad_features(tx_feat, group_col="category", value_col="discount_pct")
    tx_feat = compute_peer_group_mad_features(tx_feat, group_col="category", value_col="unit_margin")

    tx_feat = apply_baseline_a_simple_rule(tx_feat, exposure_threshold=100.0)
    tx_feat = apply_baseline_b_composite_rules(tx_feat, min_shortfall=25.0)

    # Load model
    print(f"Loading trained classifier from {model_path}...")
    clf = LapeLeakClassifier.load(str(model_path))

    # Feature matrix
    X, _ = prepare_feature_matrix(tx_feat, feature_cols=clf.feature_names)
    y_true = tx_meta["expected_detection"].to_numpy().astype(int)
    groups = tx_meta["source_record_id"].to_numpy()

    # Partition by source_record_id to evaluate strictly on held-out test partition
    from sklearn.model_selection import train_test_split
    unique_groups = np.unique(groups)
    _, test_groups = train_test_split(unique_groups, test_size=0.25, random_state=42)
    test_mask = np.isin(groups, test_groups)

    X_held_out = X[test_mask]
    y_held_out = y_true[test_mask]
    test_df = tx_feat.iloc[test_mask].copy().reset_index(drop=True)
    test_meta = tx_meta.iloc[test_mask].copy().reset_index(drop=True)

    preds = clf.predict(X_held_out, threshold=0.5)
    scores = clf.predict_proba(X_held_out)[:, 1]

    # Benchmark on held-out split
    print(f"\nRunning Evaluation Benchmark on strictly held-out test set ({len(y_held_out)} records, 0 source overlap with train):")
    results = benchmark_baselines_vs_model(
        test_df=test_df,
        test_meta=test_meta,
        y_test=y_held_out,
        model_preds=preds,
        model_scores=scores,
        budget_k=10
    )

    comp_df = pd.DataFrame(results["comparison_table"])
    print("\n" + comp_df.to_string(index=False))

    print("\nPer-Scenario-Family Breakdown on Held-Out Test Data:")
    for fam, d in results["model_family_breakdown"].items():
        if d["type"] == "control":
            print(f"  - [Control] {fam:<30}: Samples={d['sample_count']}, Clean Pass={d['clean_pass_rate']*100:.1f}%, FP={d['false_positives']}")
        else:
            print(f"  - [Leak]    {fam:<30}: Samples={d['sample_count']}, Detection Recall={d['detection_pct']}% (TP={d['true_positives']})")

    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
