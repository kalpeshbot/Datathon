"""
LAPE (Leak Analysis & Prediction Engine)
Module: Feature Engineering & Financial Equations
DATATHON 2026 Track 4: The Money Leak

Strictly adheres to:
- Explicit financial formulas
- Status segregation (Completed, Cancelled, Returned, Pending)
- Robust statistics (Median, MAD) with zero-division and small-sample safeguards
- Non-overlapping exposure calculation (no double counting)
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def compute_pricing_features(
    transactions_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes pricing, discount, margin, and shortfall equations.
    Preserves exact formulas:
      effective_unit_price = unit_price * (1.0 - discount_pct)
      unit_margin = effective_unit_price - unit_cost
      price_cost_shortfall = max(0, unit_cost - effective_unit_price)
      potential_shortfall_amount = quantity * price_cost_shortfall
    """
    tx = transactions_df.copy()
    
    # Merge with product catalog to get category, unit_cost, and list_price
    prod_cols = ["product_id", "category", "unit_cost", "list_price", "supplier_lead_days"]
    for c in ["category", "unit_cost", "list_price", "supplier_lead_days"]:
        if c in tx.columns and tx[c].isna().all():
            tx = tx.drop(columns=[c])
    cols_to_add = [c for c in prod_cols if c not in tx.columns or c == "product_id"]
    if len(cols_to_add) > 1:
        merged = tx.merge(products_df[cols_to_add], on="product_id", how="left")
    else:
        merged = tx
    
    # Ensure numerical types
    merged["unit_price"] = pd.to_numeric(merged["unit_price"], errors="coerce").fillna(0.0)
    merged["quantity"] = pd.to_numeric(merged["quantity"], errors="coerce").fillna(0.0)
    merged["discount_pct"] = pd.to_numeric(merged["discount_pct"], errors="coerce").fillna(0.0)
    merged["unit_cost"] = pd.to_numeric(merged["unit_cost"], errors="coerce").fillna(0.0)
    merged["list_price"] = pd.to_numeric(merged["list_price"], errors="coerce").fillna(0.0)

    # 1. Effective unit price
    merged["effective_unit_price"] = merged["unit_price"] * (1.0 - merged["discount_pct"])
    
    # 2. Unit margin
    merged["unit_margin"] = merged["effective_unit_price"] - merged["unit_cost"]
    
    # 3. Margin percentage (safe division)
    merged["margin_pct"] = np.where(
        merged["effective_unit_price"] > 0,
        merged["unit_margin"] / merged["effective_unit_price"],
        0.0
    )
    
    # 4. Price-cost shortfall (per unit)
    merged["price_cost_shortfall"] = np.maximum(0.0, merged["unit_cost"] - merged["effective_unit_price"])
    
    # 5. Potential shortfall amount (line item total)
    merged["potential_shortfall_amount"] = merged["quantity"] * merged["price_cost_shortfall"]
    
    # 6. Gross revenue and Net realized revenue
    merged["gross_revenue"] = merged["quantity"] * merged["unit_price"]
    merged["discount_amount"] = merged["gross_revenue"] * merged["discount_pct"]
    merged["net_revenue"] = merged["gross_revenue"] - merged["discount_amount"]
    merged["total_cogs"] = merged["quantity"] * merged["unit_cost"]
    
    # 7. List-to-sale price ratio (catalog adherence)
    merged["price_to_list_ratio"] = np.where(
        merged["list_price"] > 0,
        merged["unit_price"] / merged["list_price"],
        1.0
    )
    
    # 8. Product inherently underwater in catalog
    merged["catalog_cost_gt_list"] = (merged["unit_cost"] > merged["list_price"]).astype(int)
    
    # 9. Realized vs Unrealized Shortfall Flag based on status
    # Completed = Realized shortfall
    # Returned = Reversal candidate / potential inventory cost leak
    # Cancelled = Averted leak
    # Pending = In-flight exposure risk
    merged["realized_shortfall"] = np.where(
        merged["status"] == "Completed",
        merged["potential_shortfall_amount"],
        0.0
    )
    merged["in_flight_shortfall"] = np.where(
        merged["status"] == "Pending",
        merged["potential_shortfall_amount"],
        0.0
    )
    merged["returned_shortfall"] = np.where(
        merged["status"] == "Returned",
        merged["potential_shortfall_amount"],
        0.0
    )
    merged["averted_shortfall"] = np.where(
        merged["status"] == "Cancelled",
        merged["potential_shortfall_amount"],
        0.0
    )

    return merged


def compute_peer_group_mad_features(
    df: pd.DataFrame,
    group_col: str = "category",
    value_col: str = "discount_pct",
    min_group_size: int = 5
) -> pd.DataFrame:
    """
    Computes peer-group median and MAD (Median Absolute Deviation) deviations.
    Handles zero MAD safely (using 1e-4 or fallback to standard deviation) and small samples.
    """
    out_df = df.copy()
    
    # Calculate group median and MAD
    grouped = out_df.groupby(group_col)[value_col]
    medians = grouped.transform("median")
    
    # Absolute deviation from median
    abs_dev = (out_df[value_col] - medians).abs()
    
    # MAD per group
    mads = abs_dev.groupby(out_df[group_col]).transform("median")
    group_counts = grouped.transform("count")
    
    # Scale MAD to approximate standard deviation for normal distribution (~1.4826)
    scaled_mad = mads * 1.4826
    
    # Safe robust z-score: if MAD is near zero or sample too small, regularize
    epsilon = 1e-4
    effective_denom = np.where(
        (scaled_mad < epsilon) | (group_counts < min_group_size),
        np.maximum(grouped.transform("std").fillna(0.05), epsilon),
        scaled_mad
    )
    
    prefix = f"{group_col}_{value_col}"
    out_df[f"{prefix}_median"] = medians.fillna(0.0)
    out_df[f"{prefix}_mad"] = mads.fillna(0.0)
    out_df[f"{prefix}_robust_z"] = ((out_df[value_col] - medians) / effective_denom).fillna(0.0)
    
    return out_df


def compute_customer_history_features(
    transactions_df: pd.DataFrame,
    customers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes customer-level behavioral and leak propensity features.
    """
    tx = transactions_df.copy()
    
    # Aggregate transaction behavior per customer
    tx_agg = tx.groupby("customer_id").agg(
        cust_total_tx=("transaction_id", "count"),
        cust_completed_tx=("status", lambda s: (s == "Completed").sum()),
        cust_returned_tx=("status", lambda s: (s == "Returned").sum()),
        cust_cancelled_tx=("status", lambda s: (s == "Cancelled").sum()),
        cust_avg_discount=("discount_pct", "mean"),
        cust_max_discount=("discount_pct", "max"),
        cust_total_qty=("quantity", "sum")
    ).reset_index()
    
    tx_agg["cust_return_rate"] = np.where(
        tx_agg["cust_total_tx"] > 0,
        tx_agg["cust_returned_tx"] / tx_agg["cust_total_tx"],
        0.0
    )
    tx_agg["cust_cancellation_rate"] = np.where(
        tx_agg["cust_total_tx"] > 0,
        tx_agg["cust_cancelled_tx"] / tx_agg["cust_total_tx"],
        0.0
    )

    cust_merged = customers_df.merge(tx_agg, on="customer_id", how="left")
    cust_merged["cust_total_tx"] = cust_merged["cust_total_tx"].fillna(0)
    cust_merged["cust_return_rate"] = cust_merged["cust_return_rate"].fillna(0.0)
    cust_merged["cust_avg_discount"] = cust_merged["cust_avg_discount"].fillna(0.0)
    
    return cust_merged


def compute_inventory_features(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    transactions_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes inventory risk features:
    - Stock vs threshold gap
    - Supplier delay exposure
    - Estimated days of cover (from actual transaction demand rate)
    - Discrepancy between physical threshold and stockout_flag
    """
    inv = inventory_df.copy()
    prod_cols = ["product_id", "category", "unit_cost", "list_price", "supplier_lead_days"]
    cols_to_add = [c for c in prod_cols if c not in inv.columns or c == "product_id"]
    if len(cols_to_add) > 1:
        merged = inv.merge(products_df[cols_to_add], on="product_id", how="left")
    else:
        merged = inv
    
    merged["stock_on_hand"] = pd.to_numeric(merged["stock_on_hand"], errors="coerce").fillna(0)
    merged["reorder_threshold"] = pd.to_numeric(merged["reorder_threshold"], errors="coerce").fillna(0)
    merged["supplier_delay_days"] = pd.to_numeric(merged["supplier_delay_days"], errors="coerce").fillna(0)
    merged["supplier_lead_days"] = pd.to_numeric(merged["supplier_lead_days"], errors="coerce").fillna(0)
    
    # 1. Stock gap
    merged["stock_threshold_gap"] = merged["stock_on_hand"] - merged["reorder_threshold"]
    merged["is_below_reorder"] = (merged["stock_threshold_gap"] < 0).astype(int)
    
    # 2. Flag discrepancy
    merged["flag_discrepancy"] = (merged["stockout_flag"] != (merged["stock_on_hand"] <= merged["reorder_threshold"])).astype(int)
    
    # 3. Holding value
    merged["holding_cost_value"] = merged["stock_on_hand"] * merged["unit_cost"]
    
    # 4. Supplier risk days
    merged["total_lead_time_days"] = merged["supplier_lead_days"] + merged["supplier_delay_days"]
    
    # 5. Estimated run-rate & days of cover if transactions available
    if transactions_df is not None:
        completed_tx = transactions_df[transactions_df["status"] == "Completed"]
        demand_per_product = completed_tx.groupby("product_id")["quantity"].sum()
        # Data spans approx 365 days in 2025
        daily_demand = demand_per_product / 365.0
        merged["daily_demand_est"] = merged["product_id"].map(daily_demand).fillna(0.1)
        merged["days_of_cover"] = np.where(
            merged["daily_demand_est"] > 0,
            merged["stock_on_hand"] / merged["daily_demand_est"],
            999.0
        )
    else:
        merged["daily_demand_est"] = np.nan
        merged["days_of_cover"] = np.nan

    return merged


def compute_campaign_features(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes marketing funnel features, integrity violations, and potential budget leak.
    """
    camp = campaigns_df.copy()
    
    camp["spend"] = pd.to_numeric(camp["spend"], errors="coerce").fillna(0.0)
    camp["impressions"] = pd.to_numeric(camp["impressions"], errors="coerce").fillna(0)
    camp["clicks"] = pd.to_numeric(camp["clicks"], errors="coerce").fillna(0)
    camp["conversions"] = pd.to_numeric(camp["conversions"], errors="coerce").fillna(0)
    
    camp["clicks_gt_impressions"] = (camp["clicks"] > camp["impressions"]).astype(int)
    camp["conversions_gt_clicks"] = (camp["conversions"] > camp["clicks"]).astype(int)
    camp["funnel_violation"] = ((camp["clicks_gt_impressions"] == 1) | (camp["conversions_gt_clicks"] == 1)).astype(int)
    
    # Performance metrics
    camp["ctr"] = np.where(camp["impressions"] > 0, camp["clicks"] / camp["impressions"], 0.0)
    camp["cvr"] = np.where(camp["clicks"] > 0, camp["conversions"] / camp["clicks"], 0.0)
    camp["cpc"] = np.where(camp["clicks"] > 0, camp["spend"] / camp["clicks"], 0.0)
    camp["cpa"] = np.where(camp["conversions"] > 0, camp["spend"] / camp["conversions"], 0.0)
    
    # Financial exposure for funnel violations:
    # If the metrics violate physics, the recorded spend has compromised measurement integrity.
    camp["compromised_spend_exposure"] = np.where(
        camp["funnel_violation"] == 1,
        camp["spend"],
        0.0
    )
    
    return camp


def build_unified_transaction_feature_matrix(
    transactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    customers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assembles full transaction feature matrix for modelling and triage.
    Enforces strict isolation from synthetic labels and scenario metadata.
    """
    # 1. Base pricing features
    tx_feat = compute_pricing_features(transactions_df, products_df)
    
    # 2. Peer group MAD deviations
    tx_feat = compute_peer_group_mad_features(tx_feat, group_col="category", value_col="discount_pct")
    tx_feat = compute_peer_group_mad_features(tx_feat, group_col="category", value_col="unit_margin")
    
    # 3. Customer behavioral context
    cust_feat = compute_customer_history_features(transactions_df, customers_df)
    cust_subset = cust_feat[["customer_id", "region", "customer_segment", "cust_return_rate", "cust_avg_discount", "cust_total_tx"]]
    tx_feat = tx_feat.merge(cust_subset, on="customer_id", how="left")
    
    # 4. Temporal features (day of week, month, hour)
    if "timestamp" in tx_feat.columns:
        dt = pd.to_datetime(tx_feat["timestamp"], errors="coerce")
        tx_feat["hour"] = dt.dt.hour.fillna(12).astype(int)
        tx_feat["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        tx_feat["month"] = dt.dt.month.fillna(1).astype(int)
    
    return tx_feat
