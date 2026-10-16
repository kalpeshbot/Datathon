"""
LAPE (Leak Analysis & Prediction Engine)
Test Suite: Data Validation & Referential Integrity
DATATHON 2026 Track 4: The Money Leak
"""

import pytest
import pandas as pd
from pathlib import Path
from src.validation import (
    validate_schema,
    validate_foreign_keys,
    validate_campaign_funnel_records,
    validate_inventory_consistency
)


@pytest.fixture
def raw_data_frames():
    raw_dir = Path("data/raw")
    return {
        "customers": pd.read_csv(raw_dir / "customers.csv"),
        "products": pd.read_csv(raw_dir / "products.csv"),
        "transactions": pd.read_csv(raw_dir / "transactions.csv"),
        "inventory": pd.read_csv(raw_dir / "inventory.csv"),
        "campaigns": pd.read_csv(raw_dir / "campaigns.csv"),
    }


def test_raw_row_counts_and_uniqueness(raw_data_frames):
    """Verifies expected row counts and zero duplicates across all datasets."""
    expected_counts = {
        "customers": 3000,
        "products": 250,
        "transactions": 18000,
        "inventory": 5000,
        "campaigns": 1200
    }
    for name, expected in expected_counts.items():
        df = raw_data_frames[name]
        assert len(df) == expected, f"{name} expected {expected} rows, got {len(df)}"
        schema_audit = validate_schema(df, name)
        assert schema_audit["duplicate_rows"] == 0, f"{name} contains duplicate rows"
        assert schema_audit["is_schema_valid"] is True, f"{name} schema invalid"


def test_referential_integrity_zero_orphans(raw_data_frames):
    """Verifies foreign key constraints with zero orphan records."""
    fk_report = validate_foreign_keys(raw_data_frames)
    for key_rel, info in fk_report.items():
        assert info["valid"] is True, f"Foreign key violation on {key_rel}: {info.get('sample_orphans')}"
        assert info["orphan_count"] == 0


def test_campaign_funnel_physics_detection(raw_data_frames):
    """Verifies detection of campaign funnel violations (clicks > impr, conv > clicks)."""
    camp_val = validate_campaign_funnel_records(raw_data_frames["campaigns"])
    total_viol = int(camp_val["has_funnel_violation"].sum())
    assert total_viol == 171, f"Expected 171 funnel violations, got {total_viol}"
    assert int(camp_val["clicks_gt_impressions"].sum()) == 84
    assert int(camp_val["conversions_gt_clicks"].sum()) == 88


def test_inventory_flag_discrepancy_detection(raw_data_frames):
    """Verifies inventory flag discrepancy identification."""
    inv_val = validate_inventory_consistency(raw_data_frames["inventory"])
    mismatches = int(inv_val["flag_discrepancy"].sum())
    assert mismatches == 1253, f"Expected 1253 inventory flag discrepancies, got {mismatches}"
