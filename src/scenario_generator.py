"""
LAPE (Leak Analysis & Prediction Engine)
Module: Financial Leak Physics Scenario Generator
DATATHON 2026 Track 4: The Money Leak

Generates controlled, reproducible synthetic scenarios grounded in business physics.
Maintains full provenance, records every mutated field, and isolates scenario
ground truth from features to prevent target leakage.
"""

from typing import Dict, List, Tuple, Any, Optional
import copy
import uuid
import numpy as np
import pandas as pd


class ScenarioGenerator:
    """
    Reproducible business scenario generator for LAPE.
    Simulates operational and financial leak physics with exact provenance metadata.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

    def generate_all_scenarios(
        self,
        transactions_df: pd.DataFrame,
        products_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        campaigns_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        n_per_family: int = 60
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates balanced scenario datasets across all 10 scenario families.
        Returns:
          - scenarios_df: Data records with features matching standard schema
          - metadata_df: Separate audit provenance and ground truth labels
        """
        records = []
        metadata_list = []
        
        # Family 1: Valid Control Scenarios (expected_detection = 0)
        rec, meta = self._generate_control_valid(transactions_df, products_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 2: Isolated Pricing - Unapproved Extreme Discount (expected_detection = 1)
        rec, meta = self._generate_pricing_extreme_discount(transactions_df, products_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 3: Pricing Glitch - Price Tampering Below Cost (expected_detection = 1)
        rec, meta = self._generate_pricing_glitch_negative_margin(transactions_df, products_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 4: Campaign Funnel Inversion (clicks > impr, conv > clicks) (expected_detection = 1)
        rec, meta = self._generate_campaign_funnel_inversion(campaigns_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 5: Campaign Budget Burn / Phantom Traffic (expected_detection = 1)
        rec, meta = self._generate_campaign_budget_burn(campaigns_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 6: Inventory Phantom Stockout (expected_detection = 1)
        rec, meta = self._generate_inventory_phantom_stockout(inventory_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 7: Inventory Unflagged Depletion with High Supplier Delay (expected_detection = 1)
        rec, meta = self._generate_inventory_unflagged_depletion(inventory_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 8: Return Abuse Cluster (expected_detection = 1)
        rec, meta = self._generate_return_abuse(transactions_df, products_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 9: Compound Multi-Signal Leak (expected_detection = 1)
        rec, meta = self._generate_compound_leak(transactions_df, products_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        # Family 10: Unusual but Valid Edge Cases (Benign Controls, expected_detection = 0)
        rec, meta = self._generate_benign_edge_cases(transactions_df, products_df, n=n_per_family)
        records.extend(rec)
        metadata_list.extend(meta)
        
        scenarios_df = pd.DataFrame(records)
        metadata_df = pd.DataFrame(metadata_list)
        
        return scenarios_df, metadata_df

    def _generate_control_valid(
        self, tx_df: pd.DataFrame, prod_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Healthy completed sales with positive margins and reasonable discounts."""
        # Find transactions where unit_price > unit_cost
        merged = tx_df.merge(prod_df[["product_id", "unit_cost", "list_price"]], on="product_id", how="left")
        healthy = merged[(merged["status"] == "Completed") & (merged["unit_price"] * (1 - merged["discount_pct"]) > merged["unit_cost"] * 1.15)]
        
        sampled = healthy.sample(n=min(n, len(healthy)), random_state=self.random_seed)
        records, meta = [], []
        
        for idx, row in sampled.iterrows():
            scenario_id = f"SCEN-CTRL-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["transaction_id"]),
                "source_domain": "transactions",
                "scenario_family": "control_valid",
                "generation_method": "stratified_subsample",
                "injected_condition": "none_unmodified_healthy_control",
                "severity": 0.0,
                "expected_detection": 0,
                "changed_fields": [],
                "original_values": {},
                "modified_values": {},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_pricing_extreme_discount(
        self, tx_df: pd.DataFrame, prod_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Injects rogue discounts (50% - 85%) on high-value products creating severe loss."""
        merged = tx_df.merge(prod_df[["product_id", "unit_cost", "list_price"]], on="product_id", how="left")
        candidates = merged[merged["status"] == "Completed"].sample(n=min(n, len(merged)), random_state=self.random_seed + 1)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-DISC-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            orig_disc = float(r["discount_pct"])
            new_disc = round(float(self.rng.uniform(0.50, 0.85)), 2)
            r["discount_pct"] = new_disc
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["transaction_id"]),
                "source_domain": "transactions",
                "scenario_family": "pricing_extreme_discount",
                "generation_method": "parameter_mutation",
                "injected_condition": f"unapproved_deep_discount_{int(new_disc*100)}pct",
                "severity": float(new_disc),
                "expected_detection": 1,
                "changed_fields": ["discount_pct"],
                "original_values": {"discount_pct": orig_disc},
                "modified_values": {"discount_pct": new_disc},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_pricing_glitch_negative_margin(
        self, tx_df: pd.DataFrame, prod_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Simulates catalog/checkout glitch where unit_price is cut below unit_cost."""
        merged = tx_df.merge(prod_df[["product_id", "unit_cost", "list_price"]], on="product_id", how="left")
        candidates = merged[merged["status"] == "Completed"].sample(n=min(n, len(merged)), random_state=self.random_seed + 2)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-GLITCH-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            orig_price = float(r["unit_price"])
            cost = float(r["unit_cost"])
            # Price set to 20-50% below cost
            undercut_factor = float(self.rng.uniform(0.40, 0.75))
            new_price = round(cost * undercut_factor, 2)
            r["unit_price"] = new_price
            r["discount_pct"] = 0.05
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["transaction_id"]),
                "source_domain": "transactions",
                "scenario_family": "pricing_glitch_negative_margin",
                "generation_method": "pricing_undercut_injection",
                "injected_condition": "unit_price_tampered_below_cost",
                "severity": round(1.0 - undercut_factor, 2),
                "expected_detection": 1,
                "changed_fields": ["unit_price", "discount_pct"],
                "original_values": {"unit_price": orig_price, "discount_pct": float(row["discount_pct"])},
                "modified_values": {"unit_price": new_price, "discount_pct": 0.05},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_campaign_funnel_inversion(
        self, camp_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Generates campaign records with inverted funnel metrics (Clicks > Impressions or Conversions > Clicks)."""
        valid_camps = camp_df[(camp_df["clicks"] <= camp_df["impressions"]) & (camp_df["conversions"] <= camp_df["clicks"])]
        candidates = valid_camps.sample(n=min(n, len(valid_camps)), random_state=self.random_seed + 3)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-CAMP-INV-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            orig_impr = int(r["impressions"])
            orig_clicks = int(r["clicks"])
            orig_conv = int(r["conversions"])
            
            # Flip either clicks > impr or conv > clicks
            if self.rng.random() > 0.5:
                # clicks > impressions
                new_clicks = int(orig_impr * self.rng.uniform(1.2, 2.5))
                new_conv = min(orig_conv, int(new_clicks * 0.1))
                r["clicks"] = new_clicks
                r["conversions"] = new_conv
                changed = ["clicks", "conversions"]
                cond = "clicks_exceed_impressions"
            else:
                # conversions > clicks
                new_conv = int(orig_clicks * self.rng.uniform(1.3, 3.0))
                r["conversions"] = new_conv
                changed = ["conversions"]
                cond = "conversions_exceed_clicks"
                
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["campaign_id"]),
                "source_domain": "campaigns",
                "scenario_family": "campaign_funnel_inversion",
                "generation_method": "funnel_violation_injection",
                "injected_condition": cond,
                "severity": 1.0,
                "expected_detection": 1,
                "changed_fields": changed,
                "original_values": {"impressions": orig_impr, "clicks": orig_clicks, "conversions": orig_conv},
                "modified_values": {"clicks": r["clicks"], "conversions": r["conversions"]},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_campaign_budget_burn(
        self, camp_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """High spend marketing campaigns with zero clicks/conversions (ad fraud / bot sink)."""
        candidates = camp_df.sample(n=min(n, len(camp_df)), random_state=self.random_seed + 4)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-CAMP-BURN-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            orig_spend = float(r["spend"])
            r["spend"] = round(float(self.rng.uniform(5000, 15000)), 2)
            r["clicks"] = int(self.rng.integers(0, 3))
            r["conversions"] = 0
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["campaign_id"]),
                "source_domain": "campaigns",
                "scenario_family": "campaign_budget_burn",
                "generation_method": "ad_spend_sink",
                "injected_condition": "high_budget_zero_conversions",
                "severity": round(r["spend"] / 10000.0, 2),
                "expected_detection": 1,
                "changed_fields": ["spend", "clicks", "conversions"],
                "original_values": {"spend": orig_spend, "clicks": row["clicks"], "conversions": row["conversions"]},
                "modified_values": {"spend": r["spend"], "clicks": r["clicks"], "conversions": 0},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_inventory_phantom_stockout(
        self, inv_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Simulates warehouse false alarm: stockout_flag = 1 while warehouse has massive surplus."""
        candidates = inv_df[inv_df["stock_on_hand"] > inv_df["reorder_threshold"] * 2].sample(
            n=min(n, len(inv_df)), random_state=self.random_seed + 5
        )
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-INV-PHANTOM-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            r["stockout_flag"] = 1
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["inventory_record_id"]),
                "source_domain": "inventory",
                "scenario_family": "inventory_phantom_stockout",
                "generation_method": "flag_inversion",
                "injected_condition": "stockout_flag_1_with_surplus_stock",
                "severity": 0.8,
                "expected_detection": 1,
                "changed_fields": ["stockout_flag"],
                "original_values": {"stockout_flag": int(row["stockout_flag"])},
                "modified_values": {"stockout_flag": 1},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_inventory_unflagged_depletion(
        self, inv_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Stock depleted to near zero, supplier delay high, but stockout_flag suppressed = 0."""
        candidates = inv_df.sample(n=min(n, len(inv_df)), random_state=self.random_seed + 6)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-INV-DEPLET-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            r["stock_on_hand"] = int(self.rng.integers(0, 3))
            r["reorder_threshold"] = 50
            r["stockout_flag"] = 0  # suppressed
            r["supplier_delay_days"] = int(self.rng.integers(15, 30))
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["inventory_record_id"]),
                "source_domain": "inventory",
                "scenario_family": "inventory_unflagged_depletion",
                "generation_method": "silent_depletion_injection",
                "injected_condition": "zero_stock_unflagged_with_high_delay",
                "severity": 0.9,
                "expected_detection": 1,
                "changed_fields": ["stock_on_hand", "stockout_flag", "supplier_delay_days"],
                "original_values": {"stock_on_hand": row["stock_on_hand"], "stockout_flag": row["stockout_flag"]},
                "modified_values": {"stock_on_hand": r["stock_on_hand"], "stockout_flag": 0},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_return_abuse(
        self, tx_df: pd.DataFrame, prod_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Customer return leak: high volume return with full discount exploitation."""
        merged = tx_df.merge(prod_df[["product_id", "unit_cost", "list_price"]], on="product_id", how="left")
        candidates = merged.sample(n=min(n, len(merged)), random_state=self.random_seed + 7)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-RET-ABUSE-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            r["status"] = "Returned"
            r["quantity"] = int(self.rng.integers(5, 12))
            r["discount_pct"] = 0.35
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["transaction_id"]),
                "source_domain": "transactions",
                "scenario_family": "return_abuse_cluster",
                "generation_method": "status_volume_mutation",
                "injected_condition": "high_volume_return_with_discount",
                "severity": 0.75,
                "expected_detection": 1,
                "changed_fields": ["status", "quantity", "discount_pct"],
                "original_values": {"status": row["status"], "quantity": row["quantity"]},
                "modified_values": {"status": "Returned", "quantity": r["quantity"]},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_compound_leak(
        self, tx_df: pd.DataFrame, prod_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """Compound multi-signal leak: deep discount + quantity spike + negative margin on completed sale."""
        merged = tx_df.merge(prod_df[["product_id", "unit_cost", "list_price"]], on="product_id", how="left")
        candidates = merged.sample(n=min(n, len(merged)), random_state=self.random_seed + 8)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-COMPOUND-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            r["status"] = "Completed"
            r["quantity"] = int(self.rng.integers(8, 20))
            r["discount_pct"] = 0.50
            # Set unit_price at or slightly above cost, so 50% discount pushes it drastically negative
            r["unit_price"] = round(float(r["unit_cost"]) * 1.1, 2)
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["transaction_id"]),
                "source_domain": "transactions",
                "scenario_family": "compound_multi_signal",
                "generation_method": "multi_factor_cascade",
                "injected_condition": "bulk_qty_heavy_discount_negative_margin",
                "severity": 1.0,
                "expected_detection": 1,
                "changed_fields": ["status", "quantity", "discount_pct", "unit_price"],
                "original_values": {"quantity": row["quantity"], "discount_pct": row["discount_pct"]},
                "modified_values": {"quantity": r["quantity"], "discount_pct": 0.50, "unit_price": r["unit_price"]},
                "random_seed": self.random_seed
            })
        return records, meta

    def _generate_benign_edge_cases(
        self, tx_df: pd.DataFrame, prod_df: pd.DataFrame, n: int
    ) -> Tuple[List[dict], List[dict]]:
        """
        Unusual but completely valid scenarios to stress-test False Positive rates.
        E.g. Large bulk purchase of low-cost item with legitimate volume discount but healthy positive margin.
        """
        merged = tx_df.merge(prod_df[["product_id", "unit_cost", "list_price"]], on="product_id", how="left")
        # Find products with low cost and high list price
        healthy = merged[(merged["list_price"] > merged["unit_cost"] * 2.5) & (merged["status"] == "Completed")]
        candidates = healthy.sample(n=min(n, len(healthy)), random_state=self.random_seed + 9)
        records, meta = [], []
        
        for idx, row in candidates.iterrows():
            scenario_id = f"SCEN-BENIGN-BULK-{uuid.uuid4().hex[:8]}"
            r = row.to_dict()
            # High quantity, reasonable discount (e.g. 25%), but healthy margin
            r["quantity"] = int(self.rng.integers(15, 30))
            r["discount_pct"] = 0.25
            r["scenario_id"] = scenario_id
            records.append(r)
            
            meta.append({
                "scenario_id": scenario_id,
                "source_record_id": str(row["transaction_id"]),
                "source_domain": "transactions",
                "scenario_family": "benign_edge_case_bulk",
                "generation_method": "volume_scale_valid_margin",
                "injected_condition": "high_volume_legitimate_promotional_sale",
                "severity": 0.0,
                "expected_detection": 0,  # Valid control!
                "changed_fields": ["quantity", "discount_pct"],
                "original_values": {"quantity": row["quantity"], "discount_pct": row["discount_pct"]},
                "modified_values": {"quantity": r["quantity"], "discount_pct": 0.25},
                "random_seed": self.random_seed
            })
        return records, meta
