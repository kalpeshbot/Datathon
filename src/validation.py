"""
LAPE (Leak Analysis & Prediction Engine)
Module: Data Validation & Quality Audit
DATATHON 2026 Track 4: The Money Leak
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


EXPECTED_SCHEMAS = {
    "customers": {
        "primary_key": "customer_id",
        "required_columns": ["customer_id", "region", "customer_segment", "signup_date", "preferred_channel"]
    },
    "products": {
        "primary_key": "product_id",
        "required_columns": ["product_id", "category", "unit_cost", "list_price", "supplier_lead_days"]
    },
    "transactions": {
        "primary_key": "transaction_id",
        "required_columns": ["transaction_id", "customer_id", "product_id", "timestamp", "quantity", "unit_price", "discount_pct", "status", "channel", "payment_method"],
        "foreign_keys": [
            ("customer_id", "customers", "customer_id"),
            ("product_id", "products", "product_id")
        ]
    },
    "inventory": {
        "primary_key": "inventory_record_id",
        "required_columns": ["inventory_record_id", "product_id", "date", "stock_on_hand", "reorder_threshold", "stockout_flag", "supplier_delay_days"],
        "foreign_keys": [
            ("product_id", "products", "product_id")
        ]
    },
    "campaigns": {
        "primary_key": "campaign_id",
        "required_columns": ["campaign_id", "date", "channel", "category", "spend", "impressions", "clicks", "conversions"]
    }
}


def validate_schema(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Validates presence of required columns, missing values, duplicates, and primary key uniqueness."""
    schema_info = EXPECTED_SCHEMAS.get(dataset_name, {})
    required_cols = schema_info.get("required_columns", [])
    pk = schema_info.get("primary_key")

    missing_cols = [c for c in required_cols if c not in df.columns]
    null_counts = df[required_cols].isna().sum().to_dict() if not missing_cols else {}
    dup_rows = int(df.duplicated().sum())

    pk_unique = True
    pk_nulls = 0
    if pk and pk in df.columns:
        pk_nulls = int(df[pk].isna().sum())
        pk_unique = bool(df[pk].is_unique)

    return {
        "dataset_name": dataset_name,
        "row_count": len(df),
        "columns_present": list(df.columns),
        "missing_required_columns": missing_cols,
        "is_schema_valid": len(missing_cols) == 0,
        "null_counts": null_counts,
        "total_nulls": sum(null_counts.values()) if null_counts else 0,
        "duplicate_rows": dup_rows,
        "primary_key": pk,
        "primary_key_unique": pk_unique,
        "primary_key_nulls": pk_nulls
    }


def validate_foreign_keys(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Validates all documented foreign key constraints across tables."""
    fk_results = {}
    
    for table_name, schema_info in EXPECTED_SCHEMAS.items():
        fks = schema_info.get("foreign_keys", [])
        if table_name not in dfs:
            continue
        child_df = dfs[table_name]

        for fk_col, parent_table, parent_pk in fks:
            key_name = f"{table_name}.{fk_col} -> {parent_table}.{parent_pk}"
            if parent_table not in dfs:
                fk_results[key_name] = {"valid": False, "error": f"Parent table {parent_table} missing"}
                continue
            parent_df = dfs[parent_table]

            if fk_col not in child_df.columns or parent_pk not in parent_df.columns:
                fk_results[key_name] = {"valid": False, "error": "FK column missing in child or parent"}
                continue

            child_keys = set(child_df[fk_col].dropna())
            parent_keys = set(parent_df[parent_pk].dropna())
            orphans = child_keys - parent_keys

            fk_results[key_name] = {
                "valid": len(orphans) == 0,
                "orphan_count": len(orphans),
                "sample_orphans": list(orphans)[:5] if orphans else []
            }

    return fk_results


def validate_campaign_funnel_records(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies campaign records violating funnel physics:
    - clicks > impressions
    - conversions > clicks
    Returns a copy of violating rows with violation flags and calculations.
    """
    df = campaigns_df.copy()
    df["clicks_gt_impressions"] = df["clicks"] > df["impressions"]
    df["conversions_gt_clicks"] = df["conversions"] > df["clicks"]
    df["has_funnel_violation"] = df["clicks_gt_impressions"] | df["conversions_gt_clicks"]

    # Safe rates with division-by-zero handling
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], np.nan)
    df["cvr"] = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"], np.nan)
    df["cpc"] = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], np.nan)
    df["cpa"] = np.where(df["conversions"] > 0, df["spend"] / df["conversions"], np.nan)

    return df


def validate_inventory_consistency(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates inventory records against standard threshold rule:
    - expected_stockout: stock_on_hand <= reorder_threshold
    Identifies discrepancies with recorded stockout_flag.
    """
    df = inventory_df.copy()
    df["expected_stockout"] = (df["stock_on_hand"] <= df["reorder_threshold"]).astype(int)
    df["flag_discrepancy"] = df["stockout_flag"] != df["expected_stockout"]
    df["stock_gap"] = df["stock_on_hand"] - df["reorder_threshold"]
    return df


def validate_all_datasets(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Runs end-to-end data validation across all 5 datasets."""
    schema_reports = {name: validate_schema(df, name) for name, df in dfs.items()}
    fk_report = validate_foreign_keys(dfs)
    
    camp_val = validate_campaign_funnel_records(dfs["campaigns"]) if "campaigns" in dfs else None
    inv_val = validate_inventory_consistency(dfs["inventory"]) if "inventory" in dfs else None

    camp_violations_count = int(camp_val["has_funnel_violation"].sum()) if camp_val is not None else 0
    inv_discrepancies_count = int(inv_val["flag_discrepancy"].sum()) if inv_val is not None else 0

    return {
        "schemas": schema_reports,
        "foreign_keys": fk_report,
        "campaign_violations_count": camp_violations_count,
        "inventory_discrepancies_count": inv_discrepancies_count,
        "overall_valid": all(s["is_schema_valid"] and s["duplicate_rows"] == 0 for s in schema_reports.values()) and all(f.get("valid", False) for f in fk_report.values())
    }
