# RFM and Cohort Analysis System

A customer segmentation and longitudinal retention analytics platform developed on transactional e-commerce data using time-based cohort tracking, quintile RFM scoring, and unsupervised K-Means clustering.

---

## Executive Summary

Customer acquisition costs in retail and e-commerce typically exceed retention costs by five to seven times. Relying solely on top-line sales metrics often obscures underlying customer churn, retention decay, and disproportionate revenue dependencies across buyer segments.

This project implements a complete, data-driven customer intelligence system using 397,884 verified transactions from 4,338 unique customers. The solution combines:
1. **Time-Based Cohort Retention Analysis:** Tracks 13 monthly customer cohorts across product lifecycles to detect retention drop-offs and behavioral decay.
2. **RFM (Recency, Frequency, Monetary) Segmentation:** Scores customers across three behavioral axes into actionable commercial segments (such as Champions, Loyal Customers, and At-Risk tiers).
3. **Unsupervised Machine Learning (K-Means):** Discovers latent behavioral clusters through log-transformation, standard scaling, and optimal cluster determination.
4. **Strategic Business Recommendations:** Formulates targeted marketing interventions to reduce early churn and maximize Customer Lifetime Value (LTV).

---

## Repository Structure

```text
Customer-Segmentation-and-Cohort-Analysis/
|-- Customer_Segmentation_and_Cohort_Analysis.ipynb  # End-to-end Python notebook with full code and outputs
|-- online_retail.csv                                # Cleaned transactional dataset (397,884 rows)
|-- README.md                                        # Comprehensive technical report and business documentation
`-- requirements.txt                                 # Environment dependencies
```

---

## Dataset Details and Preprocessing

The analysis is conducted on transactional records from an online retail platform operating between 01/12/2010 and 09/12/2011.

### Raw Data Attributes:
- `InvoiceNo`: 6-digit transaction identifier. Codes starting with 'C' indicate cancellations.
- `StockCode`: 5-digit product code.
- `Description`: Product name.
- `Quantity`: Number of units purchased per transaction line.
- `InvoiceDate`: Timestamp of invoice generation.
- `UnitPrice`: Unit price per item in GBP.
- `CustomerID`: 5-digit unique customer identifier.
- `Country`: Customer residency country.

### Preprocessing and Data Cleaning Pipeline:
1. **Missing Value Handling:** Removed 135,080 records lacking a `CustomerID` (guest checkouts without persistent customer tracking).
2. **Cancellation Filtering:** Excluded 8,905 cancelled invoices identified by an `InvoiceNo` prefix of `'C'` to eliminate negative and distortionary monetary values.
3. **Non-Positive Value Removal:** Filtered out records with non-positive quantities (`Quantity <= 0`) or non-positive unit prices (`UnitPrice <= 0`).
4. **Feature Engineering:** Calculated line-item gross transaction value:
   `TotalPrice = Quantity * UnitPrice`
5. **Final Cleaned Dimensions:** 397,884 verified line-item transactions spanning 4,338 individual customers across 37 countries.

---

## Cohort Analysis Methodology and Retention Matrix

### Theoretical Framework
Cohorts are groups of customers sharing an identical initial experience within a defined time frame:
- **Time Cohorts:** Customers grouped by their acquisition date (e.g., month of first purchase).
- **Behavior Cohorts:** Customers categorized by specific service levels or historical product interaction.
- **Size Cohorts:** Customers partitioned by spend tier within their initial onboarding window.

This project focuses on **Time Cohorts** at monthly granularity.

### Analytical Procedure:
1. **Acquisition Cohort Assignment:** For each customer, identify the minimum `InvoiceDate` truncated to month level (`CohortMonth`).
2. **Transaction Month Truncation:** Truncate each subsequent transaction timestamp to month level (`InvoiceMonth`).
3. **Cohort Index Calculation:** Calculate the elapsed months between acquisition and subsequent transaction:
   `CohortIndex = (Year_diff * 12) + Month_diff + 1`
4. **Cohort Aggregation:** Group by `CohortMonth` and `CohortIndex`, counting unique `CustomerID` instances.
5. **Retention Rate Normalization:** Divide each period count by the initial cohort size (`CohortIndex = 1`) to generate proportional retention rates.

### Monthly Retention Rates Matrix (Percentages):

| Cohort Month | Initial Size | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 | Month 7 | Month 8 | Month 9 | Month 10 | Month 11 | Month 12 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2010-12** | 885 | 100% | 36.6% | 32.3% | 38.4% | 36.3% | 39.8% | 36.2% | 34.9% | 35.4% | 39.5% | 37.4% | 50.3% | 26.6% |
| **2011-01** | 417 | 100% | 22.1% | 26.6% | 23.0% | 32.1% | 28.8% | 26.1% | 24.7% | 31.2% | 34.5% | 36.7% | 15.1% | - |
| **2011-02** | 380 | 100% | 18.7% | 18.7% | 28.4% | 27.1% | 24.7% | 26.1% | 27.9% | 25.8% | 31.3% | 9.2% | - | - |
| **2011-03** | 452 | 100% | 15.0% | 25.2% | 19.9% | 22.3% | 16.8% | 26.8% | 23.0% | 27.9% | 8.6% | - | - | - |
| **2011-04** | 300 | 100% | 21.3% | 20.3% | 21.0% | 19.7% | 22.7% | 21.7% | 26.0% | 7.3% | - | - | - | - |
| **2011-05** | 284 | 100% | 19.0% | 17.3% | 17.3% | 20.8% | 23.2% | 26.4% | 9.5% | - | - | - | - | - |
| **2011-06** | 242 | 100% | 17.4% | 14.5% | 23.1% | 23.6% | 31.8% | 9.9% | - | - | - | - | - | - |
| **2011-07** | 188 | 100% | 18.1% | 18.6% | 23.9% | 27.1% | 11.2% | - | - | - | - | - | - | - |
| **2011-08** | 169 | 100% | 20.7% | 21.3% | 24.3% | 12.4% | - | - | - | - | - | - | - | - |
| **2011-09** | 299 | 100% | 23.4% | 29.8% | 12.0% | - | - | - | - | - | - | - | - | - |
| **2011-10** | 358 | 100% | 24.0% | 11.5% | - | - | - | - | - | - | - | - | - | - |
| **2011-11** | 323 | 100% | 11.1% | - | - | - | - | - | - | - | - | - | - | - |
| **2011-12** | 41 | 100% | - | - | - | - | - | - | - | - | - | - | - | - |

### Key Cohort Insights:
- **First-Month Drop-off:** Average Month-1 retention rate is **20.62%**. Across all cohorts, roughly 75% to 80% of customers do not make a second purchase in the immediately following calendar month.
- **December 2010 Core Cohort:** The inaugural cohort exhibits the strongest baseline loyalty, sustaining 32% to 50% retention throughout the year.
- **Holiday Surge:** All active cohorts display elevated repurchase activity during Month 11 (November), corresponding with seasonal holiday sales volume.

---

## RFM Segmentation Methodology and Quantitative Results

### Metric Definitions:
Relative to an operational reference cutoff date (2011-12-10):
- **Recency (R):** Number of days elapsed since the customer's most recent completed order.
  `Recency = Reference_Date - Max(Customer_InvoiceDate)`
- **Frequency (F):** Total number of distinct completed transactions made by the customer.
  `Frequency = Count(Distinct InvoiceNo)`
- **Monetary Value (M):** Total cumulative financial revenue generated by the customer.
  `Monetary = Sum(TotalPrice)`

### Quintile Scoring System:
Customers are scored from 1 to 5 across each dimension:
- **Recency:** Score 5 assigned to lowest elapsed days (most recent); Score 1 to oldest.
- **Frequency:** Score 5 assigned to highest order count; Score 1 to lowest.
- **Monetary:** Score 5 assigned to highest total revenue; Score 1 to lowest.

`RFM_Score = (R_Score * 100) + (F_Score * 10) + M_Score`

### Customer Segment Breakdown:

| Customer Segment | Customer Count | Percentage | Recency Mean (Days) | Frequency Mean (Orders) | Monetary Mean (GBP) | Segment Behavior |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Champions** | 1,139 | 26.3% | 13.3 | 10.0 | 4,960.50 | Bought recently, buy frequently, and spend the most |
| **Loyal Customers** | 821 | 18.9% | 38.0 | 3.6 | 1,480.10 | Regular buyers with consistent historical engagement |
| **At Risk** | 717 | 16.5% | 215.9 | 2.8 | 1,020.30 | High previous value but have not purchased in over 7 months |
| **Customers Needing Attention** | 615 | 14.2% | 98.4 | 1.7 | 480.20 | Average recency and spend; vulnerable to competitor attrition |
| **New Customers** | 312 | 7.2% | 15.2 | 1.1 | 390.40 | Recent initial purchase with low transaction frequency |
| **Promising / Potential Loyalists** | 415 | 9.6% | 42.1 | 1.9 | 620.80 | Recent buyers with above-average initial spend |
| **Hibernating / Lost** | 319 | 7.3% | 290.4 | 1.1 | 240.10 | Dormant accounts with low frequency and low spend |
| **Total** | **4,338** | **100.0%** | **92.5** | **4.3** | **2,054.30** | Entire analyzed customer base |

---

## Machine Learning Clustering (K-Means)

To validate the heuristic RFM quintile segments, unsupervised machine learning was executed on the normalized feature space.

### Preprocessing for Clustering:
1. **Skewness Treatment:** Applied logarithmic transformation `np.log1p` on Recency, Frequency, and Monetary attributes to correct strong right-skewness.
2. **Feature Standardization:** Scaled log-transformed features using `StandardScaler` to produce zero-mean, unit-variance vectors.

### Optimal Cluster Selection:
- **Elbow Method:** Measured Inertia (Within-Cluster Sum of Squares) from $k=2$ to $k=8$, locating an inflection elbow at $k=4$.
- **Silhouette Coefficient:** Evaluated cluster cohesion and separation, confirming $k=4$ as the natural grouping structure.

### K-Means Cluster Characteristics (k = 4):

| Cluster ID | Assigned Persona | Customer Count | Share (%) | Recency Mean | Frequency Mean | Monetary Mean (GBP) | Primary Focus |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **1** | VIP High-Value Customers | 716 | 16.5% | 12.1 days | 13.7 orders | 8,074.27 | Retention & Exclusive Access |
| **0** | Recent Low-Spend Buyers | 837 | 19.3% | 18.1 days | 2.1 orders | 551.82 | Conversion & Basket Size |
| **2** | Moderate-Value Occasional | 1,173 | 27.0% | 71.1 days | 4.1 orders | 1,802.83 | Re-engagement & Frequency |
| **3** | Inactive Low-Value Buyers | 1,612 | 37.2% | 182.5 days | 1.3 orders | 343.45 | Low-Cost Reactivation |

---

## Actionable Business Insights and Strategic Recommendations

### 1. Retention Intervention for Month-1 Churn (The 20% Cliff)
- **Observation:** Cohort analysis proves that roughly 80% of newly acquired customers fail to place an order in Month 1.
- **Action:** Implement an automated Day-14 post-purchase onboarding email sequence offering personalized accessory recommendations and a time-limited 15% discount on the second transaction.

### 2. Protecting the Core "Champions" and "VIP Cluster"
- **Observation:** Champions generate over 60% of total commercial revenue despite comprising only 26% of customer volume.
- **Action:** Establish a dedicated loyalty tier with early product launch access, priority fulfillment, and zero-threshold free shipping to reinforce brand affinity and prevent churn.

### 3. Automated Re-Activation for "At-Risk" Customers
- **Observation:** 717 customers previously spent an average of 1,020 GBP but have been dormant for over 200 days.
- **Action:** Trigger automated win-back workflows featuring dynamic "We miss you" messaging, personalized product updates based on past category purchases, and significant re-activation incentives.

### 4. Transitioning "Recent Low-Spend Buyers" into High-Frequency Buyers
- **Observation:** Cluster 0 customers display high recency (18 days) but low basket size (551 GBP).
- **Action:** Introduce volume-based discounts (such as "Spend 75 GBP, Save 15 GBP") and bundles to elevate Average Order Value (AOV).

---

## Installation and Execution Guide

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/jayakumarjk2007/Customer-Segmentation-and-Cohort-Analysis.git
cd Customer-Segmentation-and-Cohort-Analysis

# 2. Install required packages
pip install -r requirements.txt

# 3. Launch the Jupyter Notebook environment
jupyter notebook
```

### Execution
Open `Customer_Segmentation_and_Cohort_Analysis.ipynb` and execute all cells sequentially (`Cell -> Run All`).

---

## Dependencies (requirements.txt)

```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
jupyter>=1.0.0
```

---

## Author
- Jayakumar P
- Customer Segmentation and Cohort Analysis Project
