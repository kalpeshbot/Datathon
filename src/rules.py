"""
LAPE (Leak Analysis & Prediction Engine)
Module: Rule-Based Baselines & Heuristic Triage
DATATHON 2026 Track 4: The Money Leak

Provides two transparent, fully explainable rule-based approaches:
  1. Baseline A: Simple Rule / Exposure Ranking (raw dollar shortfall threshold)
  2. Baseline B: Derived-Feature Composite Rules (economic physics + statistical peer-group deviation)
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def apply_baseline_a_simple_rule(
    df: pd.DataFrame,
    exposure_threshold: float = 100.0
) -> pd.DataFrame:
    """
    Baseline A: Simple Exposure Ranking.
    Flags transactions where potential_shortfall_amount >= exposure_threshold.
    Score is normalized based on dollar exposure.
    """
    out = df.copy()
    if "potential_shortfall_amount" not in out.columns:
        if "effective_unit_price" in out.columns and "unit_cost" in out.columns:
            shortfall = np.maximum(0.0, out["unit_cost"] - out["effective_unit_price"])
            out["potential_shortfall_amount"] = out["quantity"] * shortfall
        else:
            out["potential_shortfall_amount"] = 0.0

    out["rule_a_flag"] = (out["potential_shortfall_amount"] >= exposure_threshold).astype(int)
    # Normalized ranking score in [0, 1]
    max_exp = out["potential_shortfall_amount"].max()
    out["rule_a_score"] = np.where(
        max_exp > 0,
        np.clip(out["potential_shortfall_amount"] / max_exp, 0.0, 1.0),
        0.0
    )
    out["rule_a_name"] = "Baseline_A_RawShortfall"
    return out


def apply_baseline_b_composite_rules(
    df: pd.DataFrame,
    min_shortfall: float = 25.0,
    mad_z_thresh: float = 1.5
) -> pd.DataFrame:
    """
    Baseline B: Derived-Feature Composite Heuristics.
    Combines:
      - Economic margin condition (effective unit price below cost)
      - Status filtering (distinguishes Completed realized exposure from Cancelled/Returned)
      - Peer-group MAD deviation (discount depth significantly above category median)
      - Catalog flaw separation (distinguishes independently generated product catalog flaw from rogue discounting)
    """
    out = df.copy()
    
    # 1. Price-cost shortfall condition
    has_shortfall = (out.get("price_cost_shortfall", 0.0) > 0.0)
    
    # 2. Significant dollar amount
    sig_amount = (out.get("potential_shortfall_amount", 0.0) >= min_shortfall)
    
    # 3. Status is Completed (realized financial loss)
    is_completed = (out.get("status", "") == "Completed")
    
    # 4. Statistical anomaly in discount (MAD robust z-score)
    z_col = "category_discount_pct_robust_z"
    stat_disc_anomaly = (out[z_col] > mad_z_thresh) if z_col in out.columns else (out.get("discount_pct", 0) > 0.20)
    
    # 5. Rogue transaction discount vs Catalog-level deficit
    # If the product catalog itself had unit_cost > list_price, it's a structural pricing bug
    # If catalog had list_price > unit_cost, but discount pushed it underwater, it's discount abuse/pricing execution bug
    is_execution_leak = (out.get("catalog_cost_gt_list", 0) == 0) & has_shortfall
    is_catalog_leak = (out.get("catalog_cost_gt_list", 0) == 1) & has_shortfall
    
    # Composite Rule:
    # High certainty leak: Completed + (Rogue execution discount OR (Significant amount + Statistical Anomaly))
    rule_b_condition = is_completed & (
        (is_execution_leak & (out.get("discount_pct", 0) > 0.15)) |
        (sig_amount & stat_disc_anomaly) |
        (out.get("potential_shortfall_amount", 0.0) >= 250.0)
    )
    
    out["rule_b_flag"] = rule_b_condition.astype(int)
    
    # Composite continuous scoring [0.0, 1.0]
    # Combines normalized shortfall (0.4) + robust z-score (0.3) + discount depth (0.3)
    exp_series = out["potential_shortfall_amount"] if "potential_shortfall_amount" in out.columns else pd.Series(0.0, index=out.index)
    max_exp = max(1.0, float(exp_series.max()))
    norm_exp = np.clip(exp_series.fillna(0.0) / max_exp, 0.0, 1.0)
    
    z_series = out[z_col] if z_col in out.columns else pd.Series(0.0, index=out.index)
    norm_z = np.clip(z_series.fillna(0.0) / 4.0, 0.0, 1.0)
    
    disc_series = out["discount_pct"] if "discount_pct" in out.columns else pd.Series(0.0, index=out.index)
    norm_disc = np.clip(disc_series.fillna(0.0) / 0.5, 0.0, 1.0)
    
    composite_score = (
        0.40 * norm_exp +
        0.30 * norm_z +
        0.30 * norm_disc
    ) * np.where(is_completed, 1.0, 0.5)
    
    out["rule_b_score"] = np.clip(composite_score.fillna(0.0), 0.0, 1.0)
    out["rule_b_name"] = "Baseline_B_CompositeHeuristic"
    
    # Diagnostic rule tags for transparency
    rule_tags = []
    for idx, row in out.iterrows():
        tags = []
        if row.get("price_cost_shortfall", 0) > 0:
            tags.append("NEGATIVE_MARGIN")
        if row.get("catalog_cost_gt_list", 0) == 1:
            tags.append("CATALOG_COST_GT_LIST")
        if row.get("discount_pct", 0) >= 0.25:
            tags.append("DEEP_DISCOUNT")
        if z_col in row and row[z_col] > mad_z_thresh:
            tags.append(f"MAD_DISC_Z_{round(row[z_col], 1)}")
        if row.get("status") != "Completed":
            tags.append(f"STATUS_{row.get('status')}")
        rule_tags.append(";".join(tags) if tags else "NORMAL")
    
    out["rule_tags"] = rule_tags
    return out


def evaluate_campaign_rules(camp_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates transparent funnel physics rules on marketing campaigns."""
    out = camp_df.copy()
    out["clicks_gt_impr"] = (out["clicks"] > out["impressions"]).astype(int)
    out["conv_gt_clicks"] = (out["conversions"] > out["clicks"]).astype(int)
    out["zero_conv_high_spend"] = ((out["conversions"] == 0) & (out["spend"] > 2500)).astype(int)
    
    out["campaign_alert_flag"] = (
        out["clicks_gt_impr"] | out["conv_gt_clicks"] | out["zero_conv_high_spend"]
    ).astype(int)
    
    out["campaign_exposure"] = np.where(
        out["campaign_alert_flag"] == 1,
        out["spend"],
        0.0
    )
    return out


def evaluate_inventory_rules(inv_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates inventory operational consistency rules."""
    out = inv_df.copy()
    # Rule 1: Phantom stockout (Flagged 1 but stock > threshold)
    out["phantom_stockout"] = ((out["stockout_flag"] == 1) & (out["stock_on_hand"] > out["reorder_threshold"])).astype(int)
    
    # Rule 2: Unflagged depletion (Flagged 0 but stock <= threshold)
    out["unflagged_depletion"] = ((out["stockout_flag"] == 0) & (out["stock_on_hand"] <= out["reorder_threshold"])).astype(int)
    
    # Rule 3: High supplier delay compounding critical stock
    out["supplier_risk"] = ((out["supplier_delay_days"] >= 14) & (out["stock_on_hand"] <= out["reorder_threshold"])).astype(int)
    
    out["inventory_alert_flag"] = (out["phantom_stockout"] | out["unflagged_depletion"] | out["supplier_risk"]).astype(int)
    return out
