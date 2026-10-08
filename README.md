# Customer Segmentation & Cohort Retention Analysis

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science project implementing **Time-Based Cohort Retention Analysis**, **RFM (Recency, Frequency, Monetary) Customer Segmentation**, and **Unsupervised Machine Learning (K-Means Clustering)** using transactional data from the Online Retail dataset.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Files](#repository-files)
- [Dataset Details & Preprocessing](#dataset-details--preprocessing)
- [Cohort Analysis Methodology & Retention Table](#cohort-analysis-methodology--retention-table)
- [RFM Segmentation Methodology & Results](#rfm-segmentation-methodology--results)
- [Machine Learning Clustering (K-Means)](#machine-learning-clustering-k-means)
- [Actionable Business Insights](#actionable-business-insights)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

In e-commerce and retail businesses, aggregate revenue metrics often obscure critical underlying customer churn, retention decay, and uneven revenue contribution across buyer tiers.

This project delivers:
1. **Cohort Analysis:** Longitudinal tracking of customer retention over time across 13 monthly cohorts, identifying retention drop-offs and customer lifetime patterns.
2. **RFM Segmentation:** Scoring 4,338 customers across Recency, Frequency, and Monetary value into 10 actionable behavioral tiers (*Champions*, *Loyal Customers*, *At Risk*, etc.).
3. **K-Means Clustering:** Data-driven persona discovery using log transformation, feature scaling, and optimal $k$ selection via Elbow and Silhouette methods.
4. **Actionable Retention Strategy:** Commercial recommendations to mitigate early churn and maximize Customer Lifetime Value (LTV).

---

## Repository Files

```text
Customer-Segmentation-and-Cohort-Analysis/
├── online_retail.csv                                  # Transactional dataset (397k cleaned records, 4.3k customers)
├── Customer_Segmentation_and_Cohort_Analysis.ipynb    # Complete end-to-end executed Jupyter Notebook
├── requirements.txt                                   # Required Python dependencies
└── README.md                                          # Project documentation and complete report
```

---

## Dataset Details & Preprocessing

The project analyzes the benchmark [UCI Online Retail Dataset](https://archive.icsuci.edu/dataset/352/online+retail) containing transactions from a UK online gifts retailer between **01/12/2010** and **09/12/2011**.

### Preprocessing Checklist:
1. **Handling Missing Values:** Dropped 135,080 rows lacking `CustomerID` (guest checkouts without persistent user identification).
2. **Filtering Cancellations:** Excluded 8,905 transactions where `InvoiceNo` started with `'C'` (credit memos/refunds).
3. **Invalid Quantities & Unit Prices:** Filtered out negative or zero `Quantity` and non-positive `UnitPrice` records.
4. **Feature Engineering:** Calculated line-item spend: $\text{TotalPrice} = \text{Quantity} \times \text{UnitPrice}$.
5. **Datetime Conversion:** Parsed `InvoiceDate` into standard datetime objects.

| Metric | Raw Dataset | Cleaned Dataset |
|---|:---:|:---:|
| **Total Rows** | 541,909 | **397,884** |
| **Unique Customers** | 4,372 | **4,338** |
| **Unique Products** | 4,070 | **3,665** |
| **Total Revenue** | - | **£8,911,407.90** |
| **Average Order Value** | - | **£480.87** |

---

## Cohort Analysis Methodology & Retention Table

### Approach:
- **`CohortMonth`:** Month of each customer's very first recorded transaction.
- **`InvoiceMonth`:** Calendar month of each transaction.
- **`CohortIndex`:** Elapsed months since acquisition:
  $$\text{CohortIndex} = (\text{InvoiceYear} - \text{CohortYear}) \times 12 + (\text{InvoiceMonth} - \text{CohortMonth}) + 1$$
- **`Retention Rate`:** Active customers in Month $i$ divided by initial cohort size $\times 100\%$.

### Retention Rate (%) by Monthly Cohort:

| Cohort | Size | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2010-12** | 885 | 100% | 36.6% | 32.3% | 38.4% | 36.3% | 39.8% | 36.2% | 35.4% | 35.4% | 39.5% | 37.3% | 50.0% |
| **2011-01** | 417 | 100% | 23.3% | 28.3% | 24.2% | 32.9% | 29.7% | 26.1% | 25.7% | 31.2% | 34.8% | 36.9% | - |
| **2011-02** | 380 | 100% | 24.7% | 19.2% | 27.9% | 26.8% | 24.7% | 25.5% | 28.2% | 25.8% | 31.3% | - | - |
| **2011-03** | 452 | 100% | 19.0% | 25.4% | 21.9% | 23.2% | 17.7% | 26.3% | 23.9% | 28.1% | - | - | - |
| **2011-04** | 300 | 100% | 22.7% | 22.0% | 21.0% | 20.7% | 23.7% | 23.0% | 26.0% | - | - | - | - |
| **2011-05** | 284 | 100% | 23.6% | 17.3% | 17.3% | 21.5% | 24.3% | 26.4% | - | - | - | - | - |
| **2011-06** | 242 | 100% | 20.7% | 18.6% | 27.3% | 24.8% | 31.4% | - | - | - | - | - | - |
| **2011-07** | 188 | 100% | 20.7% | 20.2% | 23.4% | 27.1% | - | - | - | - | - | - | - |
| **2011-08** | 169 | 100% | 25.4% | 25.4% | 25.4% | - | - | - | - | - | - | - | - |
| **2011-09** | 299 | 100% | 29.8% | 32.8% | - | - | - | - | - | - | - | - | - |
| **2011-10** | 358 | 100% | 26.5% | - | - | - | - | - | - | - | - | - | - |
| **2011-11** | 323 | 100% | 13.3% | - | - | - | - | - | - | - | - | - | - |

**Key Finding:** Across all cohorts, retention drops sharply in Month 1 (averaging **20.62%**), meaning ~79% of customers do not make an immediate repeat purchase. However, the customers who survive past Month 1 stabilize into a loyal baseline (~22–36% repeat purchase rate).

---

## RFM Segmentation Methodology & Results

### Scoring Logic:
- **Recency ($R$):** Days since customer's last order (quantiles 1–5, inverted so 5 = most recent).
- **Frequency ($F$):** Total count of distinct orders/invoices (rank-based quantiles 1–5).
- **Monetary ($M$):** Total cumulative spend (quantiles 1–5).

### Segment Characteristics & Revenue Contribution:

| Customer Segment | Criteria ($R, F$) | Customers | % Cust Base | % Revenue | Total Spend (£) | Mean Spend (£) | Avg Recency | Avg Orders |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champions** | $R \in [4,5], F \in [4,5]$ | 1,139 | **26.3%** | **66.5%** | £5,927,725 | **£5,204.32** | 16.3 days | 10.9 |
| **Loyal Customers** | $R \in [3,5], F \in [3,5]$ | 821 | **18.9%** | **15.2%** | £1,355,741 | £1,651.33 | 55.4 days | 4.1 |
| **At Risk** | $R \in [1,2], F \in [2,5]$ | 717 | **16.5%** | **8.1%** | £723,086 | £1,008.49 | 179.2 days | 2.9 |
| **Needing Attention** | $R \in [2,3], F \in [2,3]$ | 615 | **14.2%** | **4.5%** | £398,799 | £648.45 | 112.5 days | 2.1 |
| **Hibernating / Lost** | $R \in [1,2], F \in [1,2]$ | 364 | **8.4%** | **2.2%** | £198,034 | £544.05 | 247.1 days | 1.1 |
| **Potential Loyalists** | $R \in [4,5], F \in [2,3]$ | 178 | **4.1%** | **1.1%** | £94,682 | £531.92 | 23.4 days | 2.0 |
| **About to Sleep** | $R \in [2,3], F \in [1,2]$ | 199 | **4.6%** | **1.0%** | £92,922 | £466.94 | 108.9 days | 1.2 |
| **Promising** | $R \in [3,4], F = 1$ | 164 | **3.8%** | **0.8%** | £68,926 | £420.28 | 45.2 days | 1.0 |
| **New Customers** | $R \in [4,5], F = 1$ | 141 | **3.2%** | **0.6%** | £51,486 | £365.14 | 15.8 days | 1.0 |

---

## Machine Learning Clustering (K-Means)

1. **Log Transformation:** Applied $\log(x + 1)$ to normalize heavily right-skewed RFM distributions.
2. **Feature Scaling:** Applied `StandardScaler` ($\mu=0, \sigma=1$).
3. **Optimal $k$:** Evaluated $k \in [2, 8]$ using Elbow Inertia and Silhouette Scores; selected **$k = 4$** ($\text{Silhouette} = 0.337$).

### Empirical Clusters ($k = 4$):
- **Cluster 1 (High-Value VIPs):** 16.5% of customers, **64.9% of revenue** (Avg Spend: **£8,074**, Avg Frequency: **13.7**).
- **Cluster 2 (Frequent Regulars):** 27.0% of customers, **23.7% of revenue** (Avg Spend: **£1,802**, Avg Frequency: **4.1**).
- **Cluster 0 (Recent Shoppers):** 19.3% of customers, 5.2% of revenue (Avg Recency: **18 days**, Avg Spend: **£551**).
- **Cluster 3 (Dormant Churn Risks):** 37.2% of customers, 6.2% of revenue (Avg Recency: **182 days**, Avg Orders: **1.3**).

---

## Actionable Business Insights

1. **Protect the Champion VIPs:** Top 26.3% of customers account for **66.5% of total revenue**. Focus marketing resources on high-touch concierge services, early product previews, and premium loyalty perks.
2. **Bridge the Month-1 Churn Cliff:** The steepest drop in customer engagement occurs between Month 0 and Month 1. Deploy automated email drip campaigns within **14 days** of initial purchase featuring personalized product recommendations and 2nd-order shipping incentives.
3. **Automate Win-Back for At-Risk Customers:** 717 customers representing **£723k in historical revenue** have not purchased in over 150 days. Send time-limited re-engagement discounts before they transition to the lost pool.

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/jayakumarjk2007/Customer-Segmentation-and-Cohort-Analysis.git
cd Customer-Segmentation-and-Cohort-Analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

Launch Jupyter to open the notebook and inspect all pre-run cells, tables, and visualizations:
```bash
jupyter notebook Customer_Segmentation_and_Cohort_Analysis.ipynb
```
Select **Kernel > Restart & Run All** to re-execute all cells top-to-bottom.

---

## Author & Acknowledgments

- **Author:** Jayakumar P
- **GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)
- **Dataset:** [UCI Machine Learning Repository: Online Retail Dataset](https://archive.icsuci.edu/dataset/352/online+retail)
