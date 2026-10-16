"""
LAPE (Leak Analysis & Prediction Engine)
Test Suite: Scenario Physics, Reproducibility & Leakage Prevention
DATATHON 2026 Track 4: The Money Leak
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.scenario_generator import ScenarioGenerator
from src.model import prepare_feature_matrix, train_leak_model_with_leakage_prevention


@pytest.fixture
def mock_datasets():
    tx_df = pd.DataFrame([
        {"transaction_id": f"TX_{i:04d}", "customer_id": f"CU_{i%10:03d}", "product_id": f"PR_{i%5:03d}",
         "quantity": 2, "unit_price": 100.0, "discount_pct": 0.10, "status": "Completed"}
        for i in range(50)
    ])
    prod_df = pd.DataFrame([
        {"product_id": f"PR_{i:03d}", "category": "Electronics", "unit_cost": 50.0, "list_price": 100.0, "supplier_lead_days": 3}
        for i in range(5)
    ])
    inv_df = pd.DataFrame([
        {"inventory_record_id": f"IN_{i:04d}", "product_id": f"PR_{i%5:03d}", "date": "2025-05-01",
         "stock_on_hand": 100, "reorder_threshold": 20, "stockout_flag": 0, "supplier_delay_days": 2}
        for i in range(50)
    ])
    camp_df = pd.DataFrame([
        {"campaign_id": f"CA_{i:04d}", "date": "2025-05-01", "channel": "Web", "category": "Electronics",
         "spend": 1000.0, "impressions": 10000, "clicks": 500, "conversions": 50}
        for i in range(50)
    ])
    cust_df = pd.DataFrame([
        {"customer_id": f"CU_{i:03d}", "region": "North", "customer_segment": "Loyal", "signup_date": "2024-01-01", "preferred_channel": "Web"}
        for i in range(10)
    ])
    return tx_df, prod_df, inv_df, camp_df, cust_df


def test_scenario_reproducibility(mock_datasets):
    """Verifies that the same random seed produces identical scenario records and metadata."""
    tx, prod, inv, camp, cust = mock_datasets
    gen1 = ScenarioGenerator(random_seed=123)
    gen2 = ScenarioGenerator(random_seed=123)

    scen1, meta1 = gen1.generate_all_scenarios(tx, prod, inv, camp, cust, n_per_family=5)
    scen2, meta2 = gen2.generate_all_scenarios(tx, prod, inv, camp, cust, n_per_family=5)

    assert len(scen1) == len(scen2)
    # Check numeric equality of perturbed columns with equal_nan=True for cross-domain columns
    assert np.allclose(scen1["quantity"].to_numpy(), scen2["quantity"].to_numpy(), equal_nan=True)
    assert meta1["injected_condition"].tolist() == meta2["injected_condition"].tolist()
    assert meta1["expected_detection"].tolist() == meta2["expected_detection"].tolist()


def test_target_leakage_prevention(mock_datasets):
    """
    Verifies that feature extraction strictly excludes any scenario metadata,
    identifiers, or target labels.
    """
    tx, prod, inv, camp, cust = mock_datasets
    gen = ScenarioGenerator(random_seed=42)
    scen, meta = gen.generate_all_scenarios(tx, prod, inv, camp, cust, n_per_family=5)

    # Attach metadata columns to test whitelist filtering
    scen_with_meta = pd.concat([scen, meta[["expected_detection", "injected_condition", "severity"]]], axis=1)

    X, feature_names = prepare_feature_matrix(scen_with_meta)

    # Prohibited columns must NEVER appear in feature_names
    forbidden = ["expected_detection", "injected_condition", "scenario_id", "severity", "source_record_id", "changed_fields"]
    for f in forbidden:
        assert f not in feature_names, f"Target leakage violation: forbidden feature '{f}' present in X!"


def test_grouped_split_isolation(mock_datasets):
    """
    Verifies that source records are strictly partitioned between train and test splits,
    preventing contamination between variants of the same source record.
    """
    tx, prod, inv, camp, cust = mock_datasets
    gen = ScenarioGenerator(random_seed=42)
    scen, meta = gen.generate_all_scenarios(tx, prod, inv, camp, cust, n_per_family=10)

    # Filter to transactions
    tx_mask = meta["source_domain"] == "transactions"
    scen_tx = scen[tx_mask].copy()
    meta_tx = meta[tx_mask].copy().reset_index(drop=True)

    clf, split_info = train_leak_model_with_leakage_prevention(
        scenarios_df=scen_tx,
        metadata_df=meta_tx,
        model_type="hist_gb",
        random_state=42,
        test_size=0.3
    )

    train_sources = set(split_info["meta_train"]["source_record_id"])
    test_sources = set(split_info["meta_test"]["source_record_id"])

    overlap = train_sources.intersection(test_sources)
    assert len(overlap) == 0, f"Split isolation failure: {len(overlap)} overlapping records in train and test!"
