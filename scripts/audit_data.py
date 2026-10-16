"""
LAPE (Leak Analysis & Prediction Engine)
Dataset Audit & Baseline Verification Script
DATATHON 2026 Track 4: The Money Leak

Performs rigorous schema validation, integrity checks, and baseline metric verification.
Outputs a structured summary report in JSON and Markdown.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def audit_datasets(data_dir: str = "data/raw") -> dict:
    data_path = Path(data_dir)
    report = {
        "engine": "LAPE (Leak Analysis & Prediction Engine)",
        "data_dir": str(data_path),
        "files_checked": {},
        "row_counts": {},
        "missing_values": {},
        "duplicate_rows": {},
        "foreign_key_integrity": {},
        "transaction_status_counts": {},
        "negative_margins": {},
        "campaign_funnel_violations": {},
        "inventory_flag_discrepancies": {},
        "summary": {}
    }

    expected_files = {
        "customers.csv": 3000,
        "products.csv": 250,
        "transactions.csv": 18000,
        "inventory.csv": 5000,
        "campaigns.csv": 1200
    }

    # 1. File existence and row counts
    dfs = {}
    for filename, exp_count in expected_files.items():
        fp = data_path / filename
        if not fp.exists():
            report["files_checked"][filename] = {"exists": False, "error": "File missing"}
            continue
        df = pd.read_csv(fp)
        dfs[filename] = df
        actual_count = len(df)
        report["files_checked"][filename] = {
            "exists": True,
            "expected_rows": exp_count,
            "actual_rows": actual_count,
            "match": actual_count == exp_count,
            "columns": list(df.columns)
        }
        report["row_counts"][filename] = actual_count
        report["missing_values"][filename] = int(df.isna().sum().sum())
        report["duplicate_rows"][filename] = int(df.duplicated().sum())

    total_rows = sum(report["row_counts"].values())
    report["summary"]["total_rows_inspected"] = total_rows
    report["summary"]["expected_total_rows"] = 27450
    report["summary"]["total_rows_match"] = (total_rows == 27450)

    # 2. Foreign Key Integrity
    if "transactions.csv" in dfs and "customers.csv" in dfs:
        tx_cust = set(dfs["transactions.csv"]["customer_id"])
        cust_ids = set(dfs["customers.csv"]["customer_id"])
        orphan_tx_cust = tx_cust - cust_ids
        report["foreign_key_integrity"]["transactions_to_customers"] = {
            "orphan_count": len(orphan_tx_cust),
            "valid": len(orphan_tx_cust) == 0
        }

    if "transactions.csv" in dfs and "products.csv" in dfs:
        tx_prod = set(dfs["transactions.csv"]["product_id"])
        prod_ids = set(dfs["products.csv"]["product_id"])
        orphan_tx_prod = tx_prod - prod_ids
        report["foreign_key_integrity"]["transactions_to_products"] = {
            "orphan_count": len(orphan_tx_prod),
            "valid": len(orphan_tx_prod) == 0
        }

    if "inventory.csv" in dfs and "products.csv" in dfs:
        inv_prod = set(dfs["inventory.csv"]["product_id"])
        prod_ids = set(dfs["products.csv"]["product_id"])
        orphan_inv_prod = inv_prod - prod_ids
        report["foreign_key_integrity"]["inventory_to_products"] = {
            "orphan_count": len(orphan_inv_prod),
            "valid": len(orphan_inv_prod) == 0
        }

    # 3. Transaction Statuses
    if "transactions.csv" in dfs:
        tx_df = dfs["transactions.csv"]
        status_counts = tx_df["status"].value_counts().to_dict()
        report["transaction_status_counts"] = {k: int(v) for k, v in status_counts.items()}

    # 4. Product cost > list price
    if "products.csv" in dfs:
        prod_df = dfs["products.csv"]
        cost_gt_list = prod_df[prod_df["unit_cost"] > prod_df["list_price"]]
        report["negative_margins"]["products_cost_gt_list"] = {
            "count": len(cost_gt_list),
            "total_products": len(prod_df),
            "percentage": round(100.0 * len(cost_gt_list) / len(prod_df), 2),
            "expected_count": 76
        }

    # 5. Effective unit price < unit cost in transactions
    if "transactions.csv" in dfs and "products.csv" in dfs:
        tx_df = dfs["transactions.csv"]
        prod_df = dfs["products.csv"]
        merged = tx_df.merge(prod_df[["product_id", "unit_cost", "list_price"]], on="product_id", how="left")
        
        # effective unit price = unit_price * (1 - discount_pct)
        merged["effective_unit_price"] = merged["unit_price"] * (1.0 - merged["discount_pct"])
        merged["unit_margin"] = merged["effective_unit_price"] - merged["unit_cost"]
        merged["is_negative_margin"] = merged["effective_unit_price"] < merged["unit_cost"]
        
        neg_margin_count = int(merged["is_negative_margin"].sum())
        completed_neg = int((merged["is_negative_margin"] & (merged["status"] == "Completed")).sum())
        
        report["negative_margins"]["transactions_effective_lt_cost"] = {
            "count": neg_margin_count,
            "total_transactions": len(tx_df),
            "expected_count": 6829,
            "completed_negative_margin_count": completed_neg,
            "max_shortfall": float((merged["unit_cost"] - merged["effective_unit_price"]).max()),
            "potential_shortfall_total_all_statuses": float(
                np.maximum(0, merged["unit_cost"] - merged["effective_unit_price"] * merged["quantity"]).sum()
            ),
            "potential_shortfall_completed_only": float(
                (np.maximum(0, merged["unit_cost"] - merged["effective_unit_price"]) * merged["quantity"] * (merged["status"] == "Completed")).sum()
            )
        }

    # 6. Campaign Funnel Integrity
    if "campaigns.csv" in dfs:
        camp_df = dfs["campaigns.csv"]
        clicks_gt_impr = camp_df["clicks"] > camp_df["impressions"]
        conv_gt_clicks = camp_df["conversions"] > camp_df["clicks"]
        any_funnel_viol = clicks_gt_impr | conv_gt_clicks
        report["campaign_funnel_violations"] = {
            "clicks_gt_impressions": int(clicks_gt_impr.sum()),
            "conversions_gt_clicks": int(conv_gt_clicks.sum()),
            "total_violating_records": int(any_funnel_viol.sum()),
            "expected_count": 171
        }

    # 7. Inventory stock/threshold flag discrepancy
    if "inventory.csv" in dfs:
        inv_df = dfs["inventory.csv"]
        # Standard expectation: stockout_flag == 1 when stock_on_hand <= reorder_threshold
        expected_flag = (inv_df["stock_on_hand"] <= inv_df["reorder_threshold"]).astype(int)
        mismatch = inv_df["stockout_flag"] != expected_flag
        report["inventory_flag_discrepancies"] = {
            "total_records": len(inv_df),
            "mismatched_records": int(mismatch.sum()),
            "expected_count": 1240,
            "flag_1_but_stock_above_threshold": int(((inv_df["stockout_flag"] == 1) & (inv_df["stock_on_hand"] > inv_df["reorder_threshold"])).sum()),
            "flag_0_but_stock_le_threshold": int(((inv_df["stockout_flag"] == 0) & (inv_df["stock_on_hand"] <= inv_df["reorder_threshold"])).sum())
        }

    return report

def generate_markdown_report(report: dict, output_path: str = "reports/audit_report.md"):
    md = []
    md.append("# LAPE (Leak Analysis & Prediction Engine) — Initial Data Audit Report")
    md.append("## DATATHON 2026 Track 4: The Money Leak\n")
    md.append("> [!NOTE]")
    md.append("> This report contains empirical verification of all 5 raw datasets. All figures are recomputed directly from the data without fabricated assumptions or extrapolations as confirmed losses.\n")
    
    md.append("### 1. Dataset Integrity & Row Counts")
    md.append("| Dataset | Expected Rows | Actual Rows | Match | Missing Cells | Duplicates |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for fname, info in report["files_checked"].items():
        md.append(f"| `{fname}` | {info.get('expected_rows', 'N/A')} | {info.get('actual_rows', 'N/A')} | {'✅ Yes' if info.get('match') else '❌ No'} | {report['missing_values'].get(fname, 0)} | {report['duplicate_rows'].get(fname, 0)} |")
    md.append(f"\n**Total Rows Inspected**: {report['summary']['total_rows_inspected']:,} (Expected: {report['summary']['expected_total_rows']:,}) — {'✅ Exact Match' if report['summary']['total_rows_match'] else '❌ Discrepancy'}\n")

    md.append("### 2. Relational & Foreign Key Integrity")
    md.append("| Relationship | Join Keys | Orphan Records | Status |")
    md.append("| :--- | :--- | :--- | :--- |")
    for rel, info in report["foreign_key_integrity"].items():
        status = "✅ Valid (0 orphans)" if info["valid"] else f"❌ {info['orphan_count']} orphans"
        md.append(f"| `{rel}` | Primary -> Foreign Key | {info['orphan_count']} | {status} |")
    md.append("")

    md.append("### 3. Transaction Status Breakdown")
    md.append("| Status | Count | Percentage | Operational Meaning |")
    md.append("| :--- | :--- | :--- | :--- |")
    total_tx = report["row_counts"].get("transactions.csv", 18000)
    for st, cnt in report["transaction_status_counts"].items():
        pct = round(100.0 * cnt / total_tx, 2)
        meaning = "Fulfilled sale; revenue realized" if st == "Completed" else ("Customer return; potential refund" if st == "Returned" else ("Aborted before fulfillment" if st == "Cancelled" else "In-flight / unconfirmed"))
        md.append(f"| **{st}** | {cnt:,} | {pct}% | {meaning} |")
    md.append("")

    md.append("### 4. Mathematical Inconsistencies & Synthetic Properties")
    prod_neg = report["negative_margins"]["products_cost_gt_list"]
    tx_neg = report["negative_margins"]["transactions_effective_lt_cost"]
    camp_viol = report["campaign_funnel_violations"]
    inv_disc = report["inventory_flag_discrepancies"]

    md.append(f"- **Products with Unit Cost > List Price**: **{prod_neg['count']}** of {prod_neg['total_products']} ({prod_neg['percentage']}%) [Expected: {prod_neg['expected_count']}]")
    md.append(f"- **Transactions with Effective Price < Unit Cost**: **{tx_neg['count']:,}** of {tx_neg['total_transactions']:,} [Expected: {tx_neg['expected_count']:,}]")
    md.append(f"  - Completed status transactions with negative margin: **{tx_neg['completed_negative_margin_count']:,}**")
    md.append(f"  - Calculated potential shortfall (Completed only): **${tx_neg['potential_shortfall_completed_only']:,.2f}**")
    md.append(f"- **Campaign Funnel Constraint Violations**: **{camp_viol['total_violating_records']}** of 1,200 [Expected: {camp_viol['expected_count']}]")
    md.append(f"  - Clicks > Impressions: {camp_viol['clicks_gt_impressions']}")
    md.append(f"  - Conversions > Clicks: {camp_viol['conversions_gt_clicks']}")
    md.append(f"- **Inventory Threshold Discrepancies**: **{inv_disc['mismatched_records']:,}** of {inv_disc['total_records']:,} [Expected: {inv_disc['expected_count']:,}]")
    md.append(f"  - Flagged stockout (1) but stock > threshold: {inv_disc['flag_1_but_stock_above_threshold']:,}")
    md.append(f"  - Unflagged (0) but stock <= threshold: {inv_disc['flag_0_but_stock_le_threshold']:,}")
    md.append("\n> [!IMPORTANT]")
    md.append("> **Synthesis Confirmation**: As confirmed in `DATASET/README.txt`, unit costs and list prices were independently generated in this educational dataset. LAPE therefore treats these negative margins as mathematical screening shortfalls and synthetic conditions rather than confirmed corporate accounting fraud.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Markdown audit report written to {output_path}")

if __name__ == "__main__":
    report = audit_datasets("data/raw")
    print(json.dumps(report, indent=2))
    generate_markdown_report(report)
    with open("reports/audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Audit JSON saved to reports/audit_report.json")
