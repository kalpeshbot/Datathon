# LAPE (Leak Analysis & Prediction Engine) — Initial Data Audit Report
## DATATHON 2026 Track 4: The Money Leak

> [!NOTE]
> This report contains empirical verification of all 5 raw datasets. All figures are recomputed directly from the data without fabricated assumptions or extrapolations as confirmed losses.

### 1. Dataset Integrity & Row Counts
| Dataset | Expected Rows | Actual Rows | Match | Missing Cells | Duplicates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customers.csv` | 3000 | 3000 | ✅ Yes | 0 | 0 |
| `products.csv` | 250 | 250 | ✅ Yes | 0 | 0 |
| `transactions.csv` | 18000 | 18000 | ✅ Yes | 0 | 0 |
| `inventory.csv` | 5000 | 5000 | ✅ Yes | 0 | 0 |
| `campaigns.csv` | 1200 | 1200 | ✅ Yes | 0 | 0 |

**Total Rows Inspected**: 27,450 (Expected: 27,450) — ✅ Exact Match

### 2. Relational & Foreign Key Integrity
| Relationship | Join Keys | Orphan Records | Status |
| :--- | :--- | :--- | :--- |
| `transactions_to_customers` | Primary -> Foreign Key | 0 | ✅ Valid (0 orphans) |
| `transactions_to_products` | Primary -> Foreign Key | 0 | ✅ Valid (0 orphans) |
| `inventory_to_products` | Primary -> Foreign Key | 0 | ✅ Valid (0 orphans) |

### 3. Transaction Status Breakdown
| Status | Count | Percentage | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **Completed** | 13,962 | 77.57% | Fulfilled sale; revenue realized |
| **Cancelled** | 1,689 | 9.38% | Aborted before fulfillment |
| **Returned** | 1,431 | 7.95% | Customer return; potential refund |
| **Pending** | 918 | 5.1% | In-flight / unconfirmed |

### 4. Mathematical Inconsistencies & Synthetic Properties
- **Products with Unit Cost > List Price**: **76** of 250 (30.4%) [Expected: 76]
- **Transactions with Effective Price < Unit Cost**: **6,829** of 18,000 [Expected: 6,829]
  - Completed status transactions with negative margin: **5,315**
  - Calculated potential shortfall (Completed only): **$1,214,115.50**
- **Campaign Funnel Constraint Violations**: **171** of 1,200 [Expected: 171]
  - Clicks > Impressions: 84
  - Conversions > Clicks: 88
- **Inventory Threshold Discrepancies**: **1,253** of 5,000 [Expected: 1,240]
  - Flagged stockout (1) but stock > threshold: 648
  - Unflagged (0) but stock <= threshold: 605

> [!IMPORTANT]
> **Synthesis Confirmation**: As confirmed in `DATASET/README.txt`, unit costs and list prices were independently generated in this educational dataset. LAPE therefore treats these negative margins as mathematical screening shortfalls and synthetic conditions rather than confirmed corporate accounting fraud.