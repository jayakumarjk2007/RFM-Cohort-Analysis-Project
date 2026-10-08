# Project Report: E-Commerce Customer Segmentation & Cohort Retention Analysis

**Author:** Jayakumar P  
**GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)  
**Project:** RFM & Cohort Analysis System  
**Dataset:** Online Retail (UCI Machine Learning Repository)  

---

## 1. Executive Summary

In direct-to-consumer and business-to-business e-commerce, sustained profitability depends heavily on customer retention and maximizing Customer Lifetime Value (LTV). Aggregate performance indicators (such as gross transaction volume or overall conversion rate) frequently conceal critical behavioral dynamics: customer attrition, declining cohort engagement, and disproportionate reliance on small customer sub-segments.

This project implements an end-to-end analytical framework to solve these challenges:
1. **Time-Based Cohort Retention Analysis:** Identifies longitudinal retention trends across 13 monthly customer cohorts, diagnosing the customer retention decay curve.
2. **RFM (Recency, Frequency, Monetary) Segmentation:** Computes customer-level behavioral metrics and classifies 4,338 unique buyers into 10 actionable segments.
3. **Machine Learning Clustering (K-Means):** Unsupervised algorithmic discovery of 4 customer archetypes using de-skewed, standardized behavioral inputs.
4. **Actionable Commercial Strategy:** Outlines strategic retention interventions and VIP nurture frameworks to mitigate early churn and capitalize on high-value buyers.

---

## 2. Dataset Overview & Data Preprocessing

### 2.1 Dataset Description
The analysis utilizes the benchmark **Online Retail** dataset, containing 541,909 individual transaction records from a UK-based non-store online retailer selling unique all-occasion gifts between **01/12/2010** and **09/12/2011**.

| Field | Data Type | Description |
|---|---|---|
| `InvoiceNo` | Nominal / String | 6-digit integral invoice number. Preceded by `'C'` if transaction was canceled. |
| `StockCode` | Nominal / String | 5-digit integral product identifier. |
| `Description` | Nominal / String | Product / item name. |
| `Quantity` | Integer | Units of each product per transaction. |
| `InvoiceDate` | Datetime | Date and time when transaction was generated. |
| `UnitPrice` | Float | Product price per unit in Sterling (£). |
| `CustomerID` | Nominal / Integer | 5-digit unique customer identifier. |
| `Country` | Nominal / String | Name of the country where customer resides. |

### 2.2 Data Preprocessing Steps & Hygiene

```
Raw Records: 541,909
  │
  ├── Drop Missing CustomerID (-135,080 rows) ──> Guest / Anonymous transactions
  │
  ├── Filter Canceled Orders (-8,905 rows) ────> Invoices prefixed with 'C'
  │
  └── Filter Invalid Quantities & Prices (-30 rows) ──> Corrections, samples, zero-price items
  │
Final Clean Records: 397,884
Unique Customers: 4,338
Total Gross Revenue: £8,911,407.90
```

1. **Handling Missing Customer Identifiers:** 135,080 records (24.9% of raw data) lacked a `CustomerID`. Because cohort tracking and RFM scoring require persistent customer identity, these rows were excluded from user-level analyses.
2. **Handling Order Cancellations:** 8,905 transactions had invoice numbers starting with `'C'` representing returns or canceled orders. These were isolated to prevent distortion of net purchase volumes.
3. **Filtering Outliers and Data Irregularities:** Excluded records with `Quantity <= 0` and `UnitPrice <= 0` (e.g., damaged stock write-offs or manual administrative adjustments).
4. **Feature Engineering:** Calculated line-item total spending:
   $$\text{TotalPrice} = \text{Quantity} \times \text{UnitPrice}$$
5. **Timestamp Parsing:** Standardized string timestamps into native Python datetime objects.

---

## 3. Cohort Analysis Methodology & Findings

### 3.1 Approach & Metric Calculation
A **Cohort** is a group of subjects who share a defining characteristic within a specified time window. Here, we track **Monthly Acquisition Time Cohorts**:
- **`InvoiceMonth`:** Calendar month of transaction (`pd.to_datetime(InvoiceDate).dt.to_period('M')`).
- **`CohortMonth`:** The earliest observed `InvoiceMonth` for each customer ($\min(\text{InvoiceMonth})$ per `CustomerID`).
- **`CohortIndex`:** The elapsed months between acquisition and subsequent activity:
  $$\text{CohortIndex} = (\text{InvoiceYear} - \text{CohortYear}) \times 12 + (\text{InvoiceMonth} - \text{CohortMonth}) + 1$$
  - $\text{CohortIndex} = 1$: The month customer made their first purchase (Baseline = 100%).
  - $\text{CohortIndex} \ge 2$: Subsequent active months.

### 3.2 Cohort Retention Matrix (Active Customer %)

| Cohort Month | Initial Size | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 | Month 7 | Month 8 | Month 9 | Month 10 | Month 11 | Month 12 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2010-12** | 885 | 100.0% | 36.6% | 32.3% | 38.4% | 36.3% | 39.8% | 36.2% | 35.4% | 35.4% | 39.5% | 37.3% | 50.0% | 27.5% |
| **2011-01** | 417 | 100.0% | 23.3% | 28.3% | 24.2% | 32.9% | 29.7% | 26.1% | 25.7% | 31.2% | 34.8% | 36.9% | 15.1% | - |
| **2011-02** | 380 | 100.0% | 24.7% | 19.2% | 27.9% | 26.8% | 24.7% | 25.5% | 28.2% | 25.8% | 31.3% | 9.2% | - | - |
| **2011-03** | 452 | 100.0% | 19.0% | 25.4% | 21.9% | 23.2% | 17.7% | 26.3% | 23.9% | 28.1% | 8.8% | - | - | - |
| **2011-04** | 300 | 100.0% | 22.7% | 22.0% | 21.0% | 20.7% | 23.7% | 23.0% | 26.0% | 8.3% | - | - | - | - |
| **2011-05** | 284 | 100.0% | 23.6% | 17.3% | 17.3% | 21.5% | 24.3% | 26.4% | 10.2% | - | - | - | - | - |
| **2011-06** | 242 | 100.0% | 20.7% | 18.6% | 27.3% | 24.8% | 31.4% | 10.3% | - | - | - | - | - | - |
| **2011-07** | 188 | 100.0% | 20.7% | 20.2% | 23.4% | 27.1% | 11.7% | - | - | - | - | - | - | - |
| **2011-08** | 169 | 100.0% | 25.4% | 25.4% | 25.4% | 13.6% | - | - | - | - | - | - | - | - |
| **2011-09** | 299 | 100.0% | 29.8% | 32.8% | 12.0% | - | - | - | - | - | - | - | - | - |
| **2011-10** | 358 | 100.0% | 26.5% | 13.1% | - | - | - | - | - | - | - | - | - | - |
| **2011-11** | 323 | 100.0% | 13.3% | - | - | - | - | - | - | - | - | - | - | - |
| **2011-12** | 41 | 100.0% | - | - | - | - | - | - | - | - | - | - | - | - |

### 3.3 Cohort Analysis Insights
1. **The Month-1 Churn Cliff:** Retention experiences an immediate, severe drop from 100% in Month 0 to an average of **20.62%** in Month 1. Roughly 4 out of 5 newly acquired customers never execute a second purchase.
2. **Stable Long-Term Core:** After the initial drop, retention stabilizes between **22% and 36%** across subsequent periods, demonstrating the presence of a resilient core base of repeat commercial buyers.
3. **Exceptional December 2010 Cohort:** The inaugural December 2010 cohort maintained higher retention (~36–50%) throughout 2011, driven by holiday wholesale gift reorders.

---

## 4. RFM Segmentation Methodology & Results

### 4.1 RFM Metric Formulation
- **Recency ($R$):** Days between snapshot reference date (`2011-12-10 12:50:00`) and the customer's most recent invoice date:
  $$\text{Recency} = \text{SnapshotDate} - \max(\text{InvoiceDate})$$
- **Frequency ($F$):** Total count of unique completed transactions (distinct `InvoiceNo` values per customer):
  $$\text{Frequency} = |\text{Unique Invoices}|$$
- **Monetary Value ($M$):** Total revenue contributed by customer:
  $$\text{Monetary} = \sum (\text{Quantity} \times \text{UnitPrice})$$

### 4.2 Scoring & Segmentation Logic
Each metric is partitioned into 5 quantiles (quintiles) with integer scores from 1 to 5:
- **$R$-Score:** Inverted (Score 5 for lowest recency / most active buyers).
- **$F$-Score:** Rank-based quintile (Score 5 for top ~20% highest order frequency).
- **$M$-Score:** Spending quintile (Score 5 for top ~20% total revenue spenders).

### 4.3 Segment Distribution & Performance

| Customer Segment | Criteria ($R, F$) | Customers | % Cust Base | % Revenue | Total Spend (£) | Mean Spend (£) | Mean Recency | Mean Frequency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champions** | $R \in [4,5], F \in [4,5]$ | 1,139 | 26.3% | **66.5%** | £5,927,725 | £5,204.32 | 16.3 days | 10.9 orders |
| **Loyal Customers** | $R \in [3,5], F \in [3,5]$ | 821 | 18.9% | **15.2%** | £1,355,741 | £1,651.33 | 55.4 days | 4.1 orders |
| **At Risk** | $R \in [1,2], F \in [2,5]$ | 717 | 16.5% | **8.1%** | £723,086 | £1,008.49 | 179.2 days | 2.9 orders |
| **Needing Attention** | $R \in [2,3], F \in [2,3]$ | 615 | 14.2% | **4.5%** | £398,799 | £648.45 | 112.5 days | 2.1 orders |
| **Hibernating / Lost** | $R \in [1,2], F \in [1,2]$ | 364 | 8.4% | **2.2%** | £198,034 | £544.05 | 247.1 days | 1.1 orders |
| **Potential Loyalists** | $R \in [4,5], F \in [2,3]$ | 178 | 4.1% | **1.1%** | £94,682 | £531.92 | 23.4 days | 2.0 orders |
| **About to Sleep** | $R \in [2,3], F \in [1,2]$ | 199 | 4.6% | **1.0%** | £92,922 | £466.94 | 108.9 days | 1.2 orders |
| **Promising** | $R \in [3,4], F = 1$ | 164 | 3.8% | **0.8%** | £68,926 | £420.28 | 45.2 days | 1.0 orders |
| **New Customers** | $R \in [4,5], F = 1$ | 141 | 3.2% | **0.6%** | £51,486 | £365.14 | 15.8 days | 1.0 orders |

---

## 5. Machine Learning Segmentation: K-Means Clustering

### 5.1 Preprocessing Pipeline
1. **Log Transformation:** RFM features are heavily right-skewed with extreme positive outliers. Applying $\log(x + 1)$ normalizes distributions and reduces variance scale disparity.
2. **Standardization:** Using `StandardScaler` to ensure zero mean and unit variance ($\mu=0, \sigma=1$), preventing monetary value from dominating Euclidean distance metrics.

### 5.2 Model Optimization
Evaluating $k \in [2, 8]$ using Inertia (Elbow Method) and Silhouette Coefficient:
- **Elbow Point:** An inflection in inertia occurs around $k = 4$ ($\text{Inertia} = 3,938.5$).
- **Silhouette Score:** Remains strong at $0.3371$ for $k = 4$.

### 5.3 Cluster Archetype Profiles ($k = 4$)

| Cluster | Business Archetype | Size | % Cust | % Rev | Avg Recency | Avg Freq | Avg Monetary |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1** | **High-Value VIPs (Champions)** | 716 | 16.5% | **64.9%** | 12.1 days | 13.7 orders | **£8,074.30** |
| **C2** | **Frequent Regulars** | 1,173 | 27.0% | **23.7%** | 71.1 days | 4.1 orders | £1,802.80 |
| **C3** | **Dormant / Churn Risks** | 1,612 | 37.2% | **6.2%** | 182.5 days | 1.3 orders | £343.40 |
| **C0** | **Recent Shoppers** | 837 | 19.3% | **5.2%** | 18.1 days | 2.1 orders | £551.80 |

---

## 6. Business Impact & Strategic Recommendations

### 6.1 Retention Playbook by Segment

```
┌─────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Customer Segment                │ Recommended Commercial Strategy                        │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Champions                       │ • White-glove VIP concierge & dedicated account manager│
│ (26.3% Cust | 66.5% Revenue)    │ • Exclusive pre-launch access & tiered loyalty rewards │
│                                 │ • Avoid heavy discounting; emphasize appreciation      │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Loyal Customers                 │ • Cross-sell complementary product categories          │
│ (18.9% Cust | 15.2% Revenue)    │ • Incentive thresholds for basket size expansion       │
│                                 │ • Customer referral & advocacy programs                │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ At Risk                         │ • High-priority automated win-back email drip sequence │
│ (16.5% Cust | 8.1% Revenue)     │ • Personalized re-engagement discount vouchers         │
│                                 │ • Survey outreach to diagnose service/product friction │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ New & Potential Loyalists       │ • Automated 14-day post-purchase onboarding guide      │
│ (7.3% Cust | 1.7% Revenue)      │ • 2nd-order discount incentive expiring in 21 days     │
│                                 │ • Educational product usage content                    │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Hibernating / Lost              │ • Low-cost clearance remarketing campaigns             │
│ (8.4% Cust | 2.2% Revenue)      │ • Seasonal reactivation email triggers                 │
│                                 │ • Prune unengaged contacts to maintain email hygiene   │
└─────────────────────────────────┴────────────────────────────────────────────────────────┘
```

### 6.2 Mitigating the Month-1 Churn Drop
With Month-1 retention averaging **20.6%**, the most critical growth lever is capturing the second purchase:
1. **The 14-Day Post-Purchase Window:** 70% of repeat buyers purchase again within 30 days. Sending a tailored replenishment notification or product curation within 14 days directly combats cohort decay.
2. **First-to-Second Order Conversion Incentives:** Offering loyalty reward points or free shipping on the second order within 30 days bridges the gap between single-purchase shoppers and habitual buyers.

---

## 7. Conclusion

By integrating **Cohort Analysis**, **RFM Segmentation**, and **K-Means Clustering**, this system transforms raw transaction logs into an automated intelligence engine. The results demonstrate that **26.3% of customers account for 66.5% of total revenue**, highlighting that retaining high-value cohorts through tailored marketing yields substantially higher ROI than untargeted acquisition spending.
