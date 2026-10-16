# Mathematical Equations & Data Models

## 1. Core Financial Pricing Equations

All calculations in LAPE are reproducible and adhere strictly to economic and accounting definitions:

### 1.1 Effective Selling Price
$$\text{effective\_unit\_price} = \text{unit\_price} \times (1.0 - \text{discount\_pct})$$
*Note: `discount_pct` is strictly scaled as a decimal fraction ($0.20 = 20\%$).*

### 1.2 Unit Margin
$$\text{unit\_margin} = \text{effective\_unit\_price} - \text{unit\_cost}$$
- Positive: Line-item generated a gross profit.
- Negative: Product was sold at an economic loss.

### 1.3 Price-Cost Shortfall
$$\text{price\_cost\_shortfall} = \max(0, \text{unit\_cost} - \text{effective\_unit\_price})$$
Clamped at zero to quantify only negative margin exposure without masking losses with profitable sales.

### 1.4 Potential Financial Exposure (Line Total)
$$\text{potential\_shortfall\_amount} = \text{quantity} \times \text{price\_cost\_shortfall}$$

---

## 2. Status Segregation Matrix

To prevent inflated or fabricated financial loss reporting, LAPE segregates line items by operational fulfillment status:

| Status | Nature of Exposure | Formula | Actionable Window |
|---|---|---|---|
| **Completed** | Realized Financial Loss | $\text{quantity} \times \text{price\_cost\_shortfall}$ | Post-facto audit & clawback |
| **Pending** | In-Flight Operational Risk | $\text{quantity} \times \text{price\_cost\_shortfall}$ | Immediate dispatch hold |
| **Returned** | Restocking & Refund Exposure | $\text{quantity} \times \text{price\_cost\_shortfall}$ | Return policy review |
| **Cancelled** | Averted Exposure | $\$0.00$ Realized | Closed / No action required |

---

## 3. Robust Statistical Deviations (MAD Z-Score)

To detect statistical pricing anomalies without distortion from extreme outliers, LAPE computes the Median Absolute Deviation (MAD):

$$\text{MAD} = \text{median}\Big(\big|x_i - \text{median}(x)\big|\Big)$$

$$\text{Robust } z = \frac{x_i - \text{median}(x)}{\max(1.4826 \times \text{MAD}, 10^{-4})}$$

Grouped by product category, this identifies discount or margin anomalies relative to natural category behavior.

---

## 4. Cross-Domain Integrity Rules

### 4.1 Campaign Funnel Physics
In digital marketing, conversions cannot exceed clicks, and clicks cannot exceed impressions:
$$\text{Funnel Violation} \iff (\text{clicks} > \text{impressions}) \lor (\text{conversions} > \text{clicks})$$
When a violation occurs, measurement integrity is broken, and associated ad spend is flagged as unverified exposure.

### 4.2 Inventory Consistency
$$\text{Discrepancy} \iff (\text{stock\_on\_hand} \le \text{reorder\_threshold} \land \text{stockout\_flag} = 0) \lor (\text{stock\_on\_hand} > \text{reorder\_threshold} \times 2 \land \text{stockout\_flag} = 1)$$
Identifies ghost stockouts and unflagged inventory exhaustion risks.
