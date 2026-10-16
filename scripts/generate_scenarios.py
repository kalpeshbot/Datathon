"""
LAPE (Leak Analysis & Prediction Engine)
Script: Generate Controlled Synthetic Scenarios
DATATHON 2026 Track 4: The Money Leak

Executes the Financial Leak Physics generator with full reproducibility.
Saves data/synthetic/lape_synthetic_scenarios.csv and data/synthetic/lape_scenario_metadata.csv.
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenario_generator import ScenarioGenerator


def main():
    print("=" * 70)
    print("LAPE: Financial Leak Physics Scenario Generation Pipeline")
    print("=" * 70)

    data_dir = Path("data/raw")
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw datasets
    print("Loading raw datasets from data/raw/...")
    tx_df = pd.read_csv(data_dir / "transactions.csv")
    prod_df = pd.read_csv(data_dir / "products.csv")
    inv_df = pd.read_csv(data_dir / "inventory.csv")
    camp_df = pd.read_csv(data_dir / "campaigns.csv")
    cust_df = pd.read_csv(data_dir / "customers.csv")
    print(f"Loaded: Transactions ({len(tx_df):,}), Products ({len(prod_df):,}), Inventory ({len(inv_df):,}), Campaigns ({len(camp_df):,}), Customers ({len(cust_df):,})")

    # 2. Initialize Scenario Generator
    seed = 42
    n_per_family = 60
    print(f"Initializing ScenarioGenerator (random_seed={seed}, target_samples_per_family={n_per_family})...")
    gen = ScenarioGenerator(random_seed=seed)

    scenarios_df, metadata_df = gen.generate_all_scenarios(
        transactions_df=tx_df,
        products_df=prod_df,
        inventory_df=inv_df,
        campaigns_df=camp_df,
        customers_df=cust_df,
        n_per_family=n_per_family
    )

    print(f"\nSuccessfully generated {len(scenarios_df)} total synthetic scenario records.")
    print("\nScenario Family Breakdown:")
    breakdown = metadata_df["scenario_family"].value_counts().to_dict()
    for fam, cnt in breakdown.items():
        is_leak = int((metadata_df[metadata_df["scenario_family"] == fam]["expected_detection"] == 1).iloc[0])
        label_str = "Leak Injection" if is_leak else "Valid Control"
        print(f"  - {fam:<35}: {cnt:>4} records [{label_str}]")

    # 3. Save datasets
    scenarios_path = output_dir / "lape_synthetic_scenarios.csv"
    metadata_path = output_dir / "lape_scenario_metadata.csv"

    scenarios_df.to_csv(scenarios_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nSaved synthetic scenarios to: {scenarios_path}")
    print(f"Saved scenario metadata to:  {metadata_path}")

    # 4. Save scenario manifest
    manifest = {
        "engine": "LAPE (Leak Analysis & Prediction Engine)",
        "random_seed": seed,
        "total_scenarios": len(scenarios_df),
        "families": list(breakdown.keys()),
        "family_counts": breakdown,
        "leak_count": int(metadata_df["expected_detection"].sum()),
        "control_count": int((metadata_df["expected_detection"] == 0).sum())
    }
    with open(output_dir / "scenario_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Scenario manifest written to data/synthetic/scenario_manifest.json")


if __name__ == "__main__":
    main()
