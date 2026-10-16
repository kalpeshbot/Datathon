"""
LAPE (Leak Analysis & Prediction Engine)
Test Suite: Financial Equations & Edge Cases
DATATHON 2026 Track 4: The Money Leak
"""

import pytest
import numpy as np
import pandas as pd
from src.features import compute_pricing_features


def test_hand_calculated_positive_margin():
    """
    Hand-calculated Case 1: Healthy Positive Margin
    unit_price = 100.0, discount_pct = 0.20, unit_cost = 70.0, quantity = 5
      effective_unit_price = 100.0 * (1 - 0.20) = 80.0
      unit_margin = 80.0 - 70.0 = +10.0
      price_cost_shortfall = max(0, 70.0 - 80.0) = 0.0
      potential_shortfall_amount = 5 * 0.0 = $0.00
      gross_revenue = 5 * 100 = 500.0
      net_revenue = 500 * 0.8 = 400.0
    """
    tx_df = pd.DataFrame([{
        "transaction_id": "TX_TEST_01",
        "customer_id": "CU001",
        "product_id": "PR001",
        "quantity": 5,
        "unit_price": 100.0,
        "discount_pct": 0.20,
        "status": "Completed"
    }])
    prod_df = pd.DataFrame([{
        "product_id": "PR001",
        "category": "Electronics",
        "unit_cost": 70.0,
        "list_price": 100.0,
        "supplier_lead_days": 5
    }])

    res = compute_pricing_features(tx_df, prod_df)
    row = res.iloc[0]

    assert np.isclose(row["effective_unit_price"], 80.0)
    assert np.isclose(row["unit_margin"], 10.0)
    assert np.isclose(row["price_cost_shortfall"], 0.0)
    assert np.isclose(row["potential_shortfall_amount"], 0.0)
    assert np.isclose(row["gross_revenue"], 500.0)
    assert np.isclose(row["net_revenue"], 400.0)
    assert np.isclose(row["realized_shortfall"], 0.0)


def test_hand_calculated_negative_margin_shortfall():
    """
    Hand-calculated Case 2: Negative Margin Shortfall
    unit_price = 100.0, discount_pct = 0.40, unit_cost = 70.0, quantity = 3
      effective_unit_price = 100.0 * (1 - 0.40) = 60.0
      unit_margin = 60.0 - 70.0 = -10.0
      price_cost_shortfall = max(0, 70.0 - 60.0) = 10.0
      potential_shortfall_amount = 3 * 10.0 = $30.00
    """
    tx_df = pd.DataFrame([{
        "transaction_id": "TX_TEST_02",
        "customer_id": "CU002",
        "product_id": "PR002",
        "quantity": 3,
        "unit_price": 100.0,
        "discount_pct": 0.40,
        "status": "Completed"
    }])
    prod_df = pd.DataFrame([{
        "product_id": "PR002",
        "category": "Home",
        "unit_cost": 70.0,
        "list_price": 100.0,
        "supplier_lead_days": 4
    }])

    res = compute_pricing_features(tx_df, prod_df)
    row = res.iloc[0]

    assert np.isclose(row["effective_unit_price"], 60.0)
    assert np.isclose(row["unit_margin"], -10.0)
    assert np.isclose(row["price_cost_shortfall"], 10.0)
    assert np.isclose(row["potential_shortfall_amount"], 30.0)
    assert np.isclose(row["realized_shortfall"], 30.0)


def test_discount_fraction_representation():
    """
    Verifies that discount_pct is strictly interpreted as a decimal fraction (0.15 = 15%).
    If 0.15 were treated as 15.0, effective price would be -1400.
    """
    tx_df = pd.DataFrame([{
        "transaction_id": "TX_DISC",
        "customer_id": "CU003",
        "product_id": "PR003",
        "quantity": 1,
        "unit_price": 100.0,
        "discount_pct": 0.15,
        "status": "Completed"
    }])
    prod_df = pd.DataFrame([{
        "product_id": "PR003",
        "category": "Sports",
        "unit_cost": 50.0,
        "list_price": 100.0,
        "supplier_lead_days": 2
    }])

    res = compute_pricing_features(tx_df, prod_df)
    row = res.iloc[0]

    assert np.isclose(row["effective_unit_price"], 85.0)
    assert row["effective_unit_price"] > 0.0


def test_transaction_status_segregation():
    """
    Verifies that non-completed statuses (Pending, Returned, Cancelled)
    do NOT register as realized shortfall.
    """
    statuses = ["Completed", "Pending", "Returned", "Cancelled"]
    records = []
    for s in statuses:
        records.append({
            "transaction_id": f"TX_{s}",
            "customer_id": "CU001",
            "product_id": "PR001",
            "quantity": 2,
            "unit_price": 50.0,
            "discount_pct": 0.50,  # effective price = 25, cost = 40 -> shortfall = 15/unit, $30 line
            "status": s
        })
    tx_df = pd.DataFrame(records)
    prod_df = pd.DataFrame([{
        "product_id": "PR001",
        "category": "Fashion",
        "unit_cost": 40.0,
        "list_price": 50.0,
        "supplier_lead_days": 3
    }])

    res = compute_pricing_features(tx_df, prod_df)

    completed_row = res[res["status"] == "Completed"].iloc[0]
    pending_row = res[res["status"] == "Pending"].iloc[0]
    returned_row = res[res["status"] == "Returned"].iloc[0]
    cancelled_row = res[res["status"] == "Cancelled"].iloc[0]

    # Completed: Realized
    assert np.isclose(completed_row["realized_shortfall"], 30.0)
    assert np.isclose(completed_row["in_flight_shortfall"], 0.0)

    # Pending: In-flight only
    assert np.isclose(pending_row["realized_shortfall"], 0.0)
    assert np.isclose(pending_row["in_flight_shortfall"], 30.0)

    # Returned: Returned shortfall only
    assert np.isclose(returned_row["realized_shortfall"], 0.0)
    assert np.isclose(returned_row["returned_shortfall"], 30.0)

    # Cancelled: Averted shortfall
    assert np.isclose(cancelled_row["realized_shortfall"], 0.0)
    assert np.isclose(cancelled_row["averted_shortfall"], 30.0)


def test_zero_and_negative_edge_cases():
    """Tests zero quantity, zero price, 100% discount, and division safeguards."""
    edge_cases = [
        {"transaction_id": "T_ZERO_QTY", "customer_id": "C1", "product_id": "P1", "quantity": 0, "unit_price": 50.0, "discount_pct": 0.0, "status": "Completed"},
        {"transaction_id": "T_ZERO_PRC", "customer_id": "C1", "product_id": "P1", "quantity": 5, "unit_price": 0.0, "discount_pct": 0.0, "status": "Completed"},
        {"transaction_id": "T_FULL_DISC", "customer_id": "C1", "product_id": "P1", "quantity": 2, "unit_price": 50.0, "discount_pct": 1.0, "status": "Completed"},
    ]
    tx_df = pd.DataFrame(edge_cases)
    prod_df = pd.DataFrame([{
        "product_id": "P1",
        "category": "Beauty",
        "unit_cost": 25.0,
        "list_price": 50.0,
        "supplier_lead_days": 1
    }])

    res = compute_pricing_features(tx_df, prod_df)
    
    # Zero quantity -> 0 potential shortfall
    assert np.isclose(res[res["transaction_id"] == "T_ZERO_QTY"]["potential_shortfall_amount"].iloc[0], 0.0)
    
    # Zero price -> shortfall is quantity * cost (5 * 25 = 125.0)
    assert np.isclose(res[res["transaction_id"] == "T_ZERO_PRC"]["potential_shortfall_amount"].iloc[0], 125.0)
    
    # 100% discount -> effective price = 0 -> shortfall is quantity * cost (2 * 25 = 50.0)
    assert np.isclose(res[res["transaction_id"] == "T_FULL_DISC"]["potential_shortfall_amount"].iloc[0], 50.0)
