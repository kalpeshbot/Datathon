"""
LAPE (Leak Analysis & Prediction Engine)
Test Suite: Evidence-Linked Triage & Evaluation Metrics
DATATHON 2026 Track 4: The Money Leak
"""

import pytest
import numpy as np
import pandas as pd
from src.triage import triage_transaction_record, triage_campaign_record, triage_inventory_record, generate_unified_alert_queue
from src.evaluation import compute_metrics


def test_triage_three_distinct_dimensions():
    """
    Verifies that triage records cleanly separate:
      1. Potential Financial Exposure ($ amount & formula)
      2. Evidence Reliability (HIGH / MEDIUM / LOW)
      3. Action Urgency (CRITICAL / HIGH / MEDIUM / LOW)
    """
    tx_series = pd.Series({
        "transaction_id": "TX_TEST_999",
        "customer_id": "CU0099",
        "product_id": "PR0099",
        "quantity": 4,
        "unit_price": 50.0,
        "discount_pct": 0.50, # eff price = 25.0
        "unit_cost": 40.0,    # shortfall = 15.0/unit -> $60.00
        "status": "Completed",
        "catalog_cost_gt_list": 0,
        "channel": "Web",
        "payment_method": "Card"
    })

    triage = triage_transaction_record(tx_series)

    # Dimension 1: Exposure
    assert triage["potential_exposure_amount"] == 60.0
    assert "quantity (4)" in triage["calculation_formula"]
    assert "$60.00" in triage["calculation_formula"]

    # Dimension 2: Reliability
    assert triage["evidence_reliability"] == "HIGH"

    # Dimension 3: Urgency
    assert triage["action_urgency"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert len(triage["suggested_verification_action"]) > 10


def test_no_double_counting_in_unified_queue():
    """
    Verifies that the unified alert queue does not generate overlapping duplicate
    exposures for identical source records.
    """
    tx_feat = pd.DataFrame([
        {"transaction_id": "TX01", "customer_id": "C1", "product_id": "P1", "quantity": 1,
         "unit_price": 50.0, "discount_pct": 0.5, "unit_cost": 40.0, "status": "Completed",
         "potential_shortfall_amount": 15.0, "catalog_cost_gt_list": 0},
        {"transaction_id": "TX02", "customer_id": "C2", "product_id": "P2", "quantity": 1,
         "unit_price": 100.0, "discount_pct": 0.0, "unit_cost": 50.0, "status": "Completed",
         "potential_shortfall_amount": 0.0, "catalog_cost_gt_list": 0}
    ])
    camp_df = pd.DataFrame([
        {"campaign_id": "CA01", "channel": "Web", "category": "Home", "spend": 500.0,
         "impressions": 1000, "clicks": 2000, "conversions": 10} # Funnel violation
    ])
    inv_df = pd.DataFrame([
        {"inventory_record_id": "IN01", "product_id": "P1", "stock_on_hand": 5, "reorder_threshold": 20,
         "stockout_flag": 0, "supplier_delay_days": 10, "unit_cost": 40.0} # Discrepancy
    ])

    queue = generate_unified_alert_queue(tx_feat, camp_df, inv_df, min_exposure=10.0)

    # Total alerts should be exactly 1 tx + 1 camp + 1 inv = 3
    assert len(queue) == 3
    # All alert IDs must be unique
    assert queue["alert_id"].nunique() == len(queue)
    # Sum of exposures must be exact sum of components without inflation
    assert np.isclose(queue["potential_exposure_amount"].sum(), 15.0 + 500.0 + (15 * 40.0))


def test_hand_calculated_metric_evaluations():
    """
    Verifies metric calculation functions on a hand-computed confusion matrix.
    Given:
      y_true = [1, 1, 1, 1, 0, 0, 0, 0]  (4 Positives, 4 Negatives)
      y_pred = [1, 1, 1, 0, 0, 0, 0, 1]  (TP=3, FN=1, TN=3, FP=1)
      y_score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.6]
    Expected:
      Precision = 3 / (3 + 1) = 0.75
      Recall = 3 / 4 = 0.75
      F1 = 0.75
      FPR on controls = 1 / 4 = 0.25
      Precision@2 = Top 2 scores are 0.9 (y=1) and 0.8 (y=1) -> 2/2 = 1.0
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.6])

    metrics = compute_metrics(y_true, y_pred, y_score, budget_k=2)

    assert np.isclose(metrics["precision"], 0.75)
    assert np.isclose(metrics["recall"], 0.75)
    assert np.isclose(metrics["f1"], 0.75)
    assert np.isclose(metrics["fpr_on_controls"], 0.25)
    assert np.isclose(metrics["precision_at_2"], 1.0)
    assert metrics["true_positives"] == 3
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 3
    assert metrics["false_negatives"] == 1
