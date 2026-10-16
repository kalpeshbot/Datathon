"""
LAPE (Leak Analysis & Prediction Engine)
Module: Evidence-Linked Financial Triage
DATATHON 2026 Track 4: The Money Leak

Enforces the Three Distinct Output Dimensions:
  1. Potential Financial Exposure ($ transparent formula, status segregated, no double counting)
  2. Evidence Reliability (HIGH, MEDIUM, LOW)
  3. Action Urgency (CRITICAL, HIGH, MEDIUM, LOW based on documented operational policy)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def triage_transaction_record(row: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a transaction record and generates an evidence-linked triage profile.
    Separates exposure, reliability, and urgency strictly.
    """
    tx_id = str(row.get("transaction_id", "UNKNOWN"))
    qty = float(row.get("quantity", 0))
    price = float(row.get("unit_price", 0))
    disc = float(row.get("discount_pct", 0))
    eff_price = float(row.get("effective_unit_price", price * (1.0 - disc)))
    cost = float(row.get("unit_cost", 0))
    status = str(row.get("status", "Unknown"))
    catalog_flaw = int(row.get("catalog_cost_gt_list", 0))
    shortfall_unit = max(0.0, cost - eff_price)
    exposure_amount = round(qty * shortfall_unit, 2)
    
    # 1. Exposure Nature
    if status == "Completed":
        exp_nature = "Realized Shortfall (Sale Fulfilled)"
    elif status == "Pending":
        exp_nature = "In-Flight Exposure (Order Pending Fulfillment)"
    elif status == "Returned":
        exp_nature = "Returned Item Cost at Risk (Refund / Restocking)"
    elif status == "Cancelled":
        exp_nature = "Averted Exposure (Order Cancelled Prior to Fulfillment)"
    else:
        exp_nature = "Unclassified Transaction Exposure"

    # 2. Formula Breakdown
    qty_disp = int(qty) if float(qty).is_integer() else qty
    formula_str = (
        f"quantity ({qty_disp}) * max(0, unit_cost (${cost:.2f}) - "
        f"[unit_price (${price:.2f}) * (1 - discount_pct ({disc*100:.1f}%)) = ${eff_price:.2f}]) "
        f"= ${exposure_amount:.2f}"
    )

    # 3. Evidence Reliability
    # High: Completed transaction where catalog is healthy (list_price >= unit_cost) and price tampers below cost
    # Medium: Synthetic dataset anomaly where product unit_cost > list_price in catalog
    # Low: Cancelled or Pending transaction where loss was not realized
    if exposure_amount == 0:
        reliability = "NONE"
    elif status == "Cancelled":
        reliability = "LOW"
    elif status == "Pending":
        reliability = "LOW"
    elif catalog_flaw == 1:
        reliability = "MEDIUM"  # Known synthetic artifact: cost > list in products.csv
    elif status == "Completed" and shortfall_unit > 0:
        reliability = "HIGH"
    else:
        reliability = "MEDIUM"

    # 4. Action Urgency (Operational Policy)
    if exposure_amount == 0:
        urgency = "LOW"
    elif status == "Cancelled":
        urgency = "LOW"  # Already cancelled, no immediate cash leak
    elif status == "Completed":
        if exposure_amount >= 500.0:
            urgency = "CRITICAL"
        elif exposure_amount >= 150.0:
            urgency = "HIGH"
        elif exposure_amount >= 25.0:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
    elif status == "Pending":
        if exposure_amount >= 300.0:
            urgency = "HIGH"  # Intervene before shipment!
        elif exposure_amount >= 100.0:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
    else:
        urgency = "LOW"

    # 5. Suggested Verification Action
    if status == "Pending" and exposure_amount > 50:
        action = f"Hold shipment for order {tx_id}. Verify coupon/override discount of {disc*100:.0f}% with sales manager before dispatch."
    elif catalog_flaw == 1:
        action = f"Catalog product {row.get('product_id')} has unit_cost (${cost:.2f}) > list_price (${row.get('list_price', 0):.2f}). Audit supplier contract and master price table."
    elif disc > 0.30:
        action = f"Audit customer {row.get('customer_id')} for discount stacking or promotional code leakage on channel {row.get('channel')}."
    elif status == "Returned":
        action = f"Inspect returned goods for product {row.get('product_id')} and review return justification to avert restocking write-off."
    else:
        action = f"Verify invoice and payment reconciliation on {row.get('payment_method')} transaction {tx_id}."

    return {
        "alert_id": f"ALT-TX-{tx_id}",
        "domain": "Pricing & Transactions",
        "source_record_id": tx_id,
        "customer_id": row.get("customer_id", "N/A"),
        "product_id": row.get("product_id", "N/A"),
        "status": status,
        "potential_exposure_amount": exposure_amount,
        "exposure_nature": exp_nature,
        "calculation_formula": formula_str,
        "evidence_reliability": reliability,
        "action_urgency": urgency,
        "suggested_verification_action": action,
        "raw_attributes": {
            "quantity": qty,
            "unit_price": price,
            "discount_pct": disc,
            "effective_unit_price": eff_price,
            "unit_cost": cost,
            "channel": row.get("channel", "N/A"),
            "payment_method": row.get("payment_method", "N/A"),
            "timestamp": str(row.get("timestamp", "N/A"))
        }
    }


def triage_campaign_record(row: pd.Series) -> Dict[str, Any]:
    """Generates triage profile for marketing campaign records."""
    camp_id = str(row.get("campaign_id", "UNKNOWN"))
    spend = float(row.get("spend", 0.0))
    impr = int(row.get("impressions", 0))
    clicks = int(row.get("clicks", 0))
    conv = int(row.get("conversions", 0))

    clicks_gt_impr = clicks > impr
    conv_gt_clicks = conv > clicks
    has_violation = clicks_gt_impr or conv_gt_clicks

    if clicks_gt_impr and conv_gt_clicks:
        viol_desc = f"Physics Inversion: clicks ({clicks:,}) > impressions ({impr:,}) AND conversions ({conv:,}) > clicks ({clicks:,})"
    elif clicks_gt_impr:
        viol_desc = f"Physics Inversion: clicks ({clicks:,}) > impressions ({impr:,})"
    elif conv_gt_clicks:
        viol_desc = f"Physics Inversion: conversions ({conv:,}) > clicks ({clicks:,})"
    else:
        viol_desc = "Standard Funnel Flow"

    exposure = round(spend, 2) if has_violation else 0.0
    formula_str = (
        f"Spend (${spend:,.2f}) flagged as compromised measurement exposure due to {viol_desc}"
        if has_violation else "Spend confirmed within physical funnel constraints"
    )

    reliability = "HIGH" if has_violation else "NONE"
    
    if not has_violation:
        urgency = "LOW"
    elif spend >= 5000.0:
        urgency = "CRITICAL"
    elif spend >= 1500.0:
        urgency = "HIGH"
    else:
        urgency = "MEDIUM"

    action = (
        f"Pause campaign {camp_id} on {row.get('channel')} ({row.get('category')}). "
        f"Audit tracking tags / UTM parameters and request ad platform reconciliation credit."
        if has_violation else "No action required."
    )

    return {
        "alert_id": f"ALT-CAMP-{camp_id}",
        "domain": "Marketing Campaigns",
        "source_record_id": camp_id,
        "channel": row.get("channel", "N/A"),
        "category": row.get("category", "N/A"),
        "potential_exposure_amount": exposure,
        "exposure_nature": "Compromised Ad Spend Exposure",
        "calculation_formula": formula_str,
        "evidence_reliability": reliability,
        "action_urgency": urgency,
        "suggested_verification_action": action,
        "raw_attributes": {
            "spend": spend,
            "impressions": impr,
            "clicks": clicks,
            "conversions": conv,
            "date": str(row.get("date", "N/A"))
        }
    }


def triage_inventory_record(row: pd.Series) -> Dict[str, Any]:
    """Generates triage profile for inventory records."""
    rec_id = str(row.get("inventory_record_id", "UNKNOWN"))
    prod_id = str(row.get("product_id", "UNKNOWN"))
    stock = int(row.get("stock_on_hand", 0))
    threshold = int(row.get("reorder_threshold", 0))
    flag = int(row.get("stockout_flag", 0))
    delay = int(row.get("supplier_delay_days", 0))
    unit_cost = float(row.get("unit_cost", 50.0))

    is_below = stock <= threshold
    discrepancy = flag != int(is_below)
    
    if flag == 1 and stock > threshold:
        nature = "Phantom Stockout Flag (Warehouse Surplus Locked)"
        exposure = round(stock * unit_cost * 0.05, 2)  # 5% working capital stall estimate
        formula_str = f"Surplus stock ({stock} units > {threshold} threshold) locked under false stockout flag * unit cost (${unit_cost:.2f}) * 5% stall factor"
        reliability = "HIGH"
        urgency = "MEDIUM"
        action = f"Reset stockout_flag to 0 for {prod_id}. Release allocated warehouse reserve to live storefront."
    elif flag == 0 and stock <= threshold:
        nature = "Unflagged Stock Depletion (Lost Sales Exposure)"
        gap = threshold - stock
        exposure = round(gap * unit_cost, 2)
        formula_str = f"Stock deficit ({gap} units below {threshold} threshold) without reorder trigger * unit cost (${unit_cost:.2f})"
        reliability = "HIGH"
        urgency = "CRITICAL" if (stock == 0 and delay > 10) else "HIGH"
        action = f"Trigger expedited purchase order for {prod_id}. Supplier delay is {delay} days; stock is critical ({stock} on hand)."
    else:
        nature = "Consistent Inventory Level"
        exposure = 0.0
        formula_str = "Stock level and stockout flag are physically consistent."
        reliability = "NONE"
        urgency = "LOW"
        action = "Maintain normal replenishment monitoring."

    return {
        "alert_id": f"ALT-INV-{rec_id}",
        "domain": "Inventory Operations",
        "source_record_id": rec_id,
        "product_id": prod_id,
        "potential_exposure_amount": exposure,
        "exposure_nature": nature,
        "calculation_formula": formula_str,
        "evidence_reliability": reliability,
        "action_urgency": urgency,
        "suggested_verification_action": action,
        "raw_attributes": {
            "stock_on_hand": stock,
            "reorder_threshold": threshold,
            "stockout_flag": flag,
            "supplier_delay_days": delay,
            "date": str(row.get("date", "N/A"))
        }
    }


def generate_unified_alert_queue(
    transactions_features_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    min_exposure: float = 10.0
) -> pd.DataFrame:
    """
    Assembles unified cross-domain alert queue without double counting.
    Each alert has clear exposure, reliability, urgency, and mathematical formula.
    """
    alerts = []

    # 1. Transactions: completed with price_cost_shortfall > 0 or high in-flight
    tx_candidates = transactions_features_df[
        (transactions_features_df["potential_shortfall_amount"] >= min_exposure)
    ]
    for idx, row in tx_candidates.iterrows():
        alerts.append(triage_transaction_record(row))

    # 2. Campaigns: funnel physics violations
    camp_candidates = campaigns_df[
        (campaigns_df["clicks"] > campaigns_df["impressions"]) |
        (campaigns_df["conversions"] > campaigns_df["clicks"])
    ]
    for idx, row in camp_candidates.iterrows():
        alerts.append(triage_campaign_record(row))

    # 3. Inventory: stockout flag discrepancies
    inv_candidates = inventory_df[
        inventory_df["stockout_flag"] != (inventory_df["stock_on_hand"] <= inventory_df["reorder_threshold"]).astype(int)
    ]
    for idx, row in inv_candidates.iterrows():
        alerts.append(triage_inventory_record(row))

    alert_df = pd.DataFrame(alerts)
    return alert_df
