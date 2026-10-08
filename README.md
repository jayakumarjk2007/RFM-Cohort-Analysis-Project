# Customer Segmentation & Cohort Retention Analysis

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science and customer intelligence system implementing **Time-Based Cohort Analysis**, **RFM (Recency, Frequency, Monetary) Customer Segmentation**, and **Unsupervised Machine Learning (K-Means Clustering)** on e-commerce transaction data.

---

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Dataset Specifications & Preprocessing](#dataset-specifications--preprocessing)
- [Cohort Analysis (Customer Retention)](#cohort-analysis-customer-retention)
- [RFM Segmentation Methodology](#rfm-segmentation-methodology)
- [Machine Learning Clustering (K-Means)](#machine-learning-clustering-k-means)
- [Key Insights & Strategic Recommendations](#key-insights--strategic-recommendations)
- [Technologies & Libraries](#technologies--libraries)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

In e-commerce and retail analytics, aggregate metrics such as total revenue or overall conversion rate often mask underlying customer attrition, shifting buyer demographics, and retention drop-offs.

This project delivers an automated intelligence framework to:
1. **Track Retention Decay:** Measure how customer cohorts retain and spend over their lifetime using longitudinal cohort matrices.
2. **Quantify Customer Value:** Score each buyer across Recency, Frequency, and Monetary dimensions to classify 4,338 customers into 10 targeted behavioral segments.
3. **Discover Natural Clusters:** Apply K-Means clustering on de-skewed, standardized behavioral telemetry to discover empirical customer personas.
4. **Drive Commercial ROI:** Provide data-driven marketing playbooks to protect high-value VIP revenue and mitigate early-stage customer churn.

---

## System Architecture

```
Raw Transaction Data (541,909 records)
       │
       ▼
Data Cleaning & Preprocessing (Missing CustomerID, Cancellations 'C', Invalid Prices/Quantities)
       │
       ▼ Cleaned Data (397,884 records | 4,338 Customers)
 ┌─────┴────────────────────────────────────────────────┐
 ▼                                                      ▼
Cohort Analysis Pipeline                               RFM & ML Segmentation Pipeline
 ├── Group by CustomerID -> CohortMonth                 ├── Compute Recency, Frequency, Monetary
 ├── Truncate InvoiceDate -> InvoiceMonth               ├── 1-5 Quantile Scoring (R, F, M)
 ├── CohortIndex = (ΔYear * 12) + ΔMonth + 1            ├── 10 Behavioral Customer Segments
 ├── Active Customer Count Pivot Matrix                 ├── Log Transform -> StandardScaler
 └── Percentage Retention Matrix & Heatmap              └── K-Means Clustering (Elbow & Silhouette)
       │                                                │
       └────────────────────────┬───────────────────────┘
                                │
                                ▼
         Actionable Commercial Retention Playbooks & VIP Strategies
```

---

## Repository Structure

```text
Customer-Segmentation-and-Cohort-Analysis/
├── data/
│   ├── online_retail.csv              # Source transactional dataset (541,909 rows)
│   └── customer_rfm_segments.csv      # Exported customer-level dataset with RFM & cluster tags
├── figures/
│   ├── eda_overview.png               # Monthly revenue trends & top products/markets
│   ├── cohort_retention_heatmap.png   # Monthly retention percentage matrix heatmap
│   ├── cohort_retention_curves.png    # Longitudinal retention decay curves
│   ├── rfm_distributions.png          # Raw vs log-transformed RFM feature distributions
│   ├── rfm_segment_summary.png        # Customer share vs revenue share by segment
│   └── kmeans_evaluation.png          # Elbow inertia curve & Silhouette scores across k
├── notebooks/
│   └── Customer_Segmentation_and_Cohort_Analysis.ipynb # Fully executed analytical notebook
├── reports/
│   └── Project_Report.md              # Detailed academic/business project report
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Ingestion, validation, and data cleaning routines
│   ├── cohort_analysis.py             # Cohort index, pivot tables, and retention matrices
│   ├── rfm_analysis.py                # RFM computation, quantile scoring & segment mapping
│   ├── kmeans_segmentation.py         # Skewness correction, scaling, K-Means & profiling
│   └── visualization.py               # Publication-ready plotting utilities
├── main.py                            # Standalone end-to-end CLI execution pipeline
├── requirements.txt                   # Project package dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Comprehensive project documentation
```

---

## Dataset Specifications & Preprocessing

The analysis uses the benchmark [UCI Online Retail Dataset](https://archive.icsuci.edu/dataset/352/online+retail), containing real-world transactions from a UK online gifts retailer between **01/12/2010** and **09/12/2011**.

### Data Cleaning Checklist:
- **Missing `CustomerID`:** Filtered 135,080 guest transactions lacking user tracking.
- **Canceled Transactions:** Filtered 8,905 invoices prefixed with `'C'` (credit memos/returns).
- **Price/Quantity Hygiene:** Filtered non-positive quantities and zero unit prices (damaged inventory write-offs).
- **Feature Engineering:** Calculated line-item spend: $\text{TotalPrice} = \text{Quantity} \times \text{UnitPrice}$.

| Metric | Raw Dataset | Cleaned Dataset |
|---|:---:|:---:|
| **Total Rows** | 541,909 | **397,884** |
| **Unique Customers** | 4,372 | **4,338** |
| **Unique Products** | 4,070 | **3,665** |
| **Total Revenue** | - | **£8,911,407.90** |
| **Average Order Value** | - | **£480.87** |

---

## Cohort Analysis (Customer Retention)

Customers are assigned to **Monthly Acquisition Cohorts** based on the month of their first recorded transaction (`CohortMonth`). Each subsequent purchase is indexed relative to acquisition:

$$\text{CohortIndex} = (\text{InvoiceYear} - \text{CohortYear}) \times 12 + (\text{InvoiceMonth} - \text{CohortMonth}) + 1$$

$$\text{Retention Rate} = \frac{\text{Active Customers in Period } i}{\text{Total Customers Acquired in Cohort}} \times 100\%$$

### Retention Heatmap Findings:
- **Month-1 Drop-off:** Retention experiences an initial drop from 100% in Month 0 to an average of **20.62%** in Month 1 across all cohorts.
- **Long-Term Baseline:** Cohorts stabilize between **22% and 36%** in subsequent months, demonstrating a resilient repeat wholesale customer core.
- **Top Performing Cohort:** The December 2010 cohort maintained the highest long-term retention (up to 50% in Month 11).

---

## RFM Segmentation Methodology

### Metric Definitions:
- **Recency ($R$):** Days elapsed between snapshot evaluation date (`2011-12-10`) and customer's latest order date.
- **Frequency ($F$):** Total count of unique completed invoices per customer.
- **Monetary ($M$):** Cumulative spending (£) per customer.

### Scoring & Segment Performance:

| Customer Segment | Criteria ($R, F$) | Customers | % Cust Base | % Revenue | Mean Spend (£) | Mean Recency | Mean Frequency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champions** | $R \in [4,5], F \in [4,5]$ | 1,139 | **26.3%** | **66.5%** | **£5,204.32** | 16.3 days | 10.9 orders |
| **Loyal Customers** | $R \in [3,5], F \in [3,5]$ | 821 | **18.9%** | **15.2%** | £1,651.33 | 55.4 days | 4.1 orders |
| **At Risk** | $R \in [1,2], F \in [2,5]$ | 717 | **16.5%** | **8.1%** | £1,008.49 | 179.2 days | 2.9 orders |
| **Needing Attention** | $R \in [2,3], F \in [2,3]$ | 615 | **14.2%** | **4.5%** | £648.45 | 112.5 days | 2.1 orders |
| **Hibernating / Lost** | $R \in [1,2], F \in [1,2]$ | 364 | **8.4%** | **2.2%** | £544.05 | 247.1 days | 1.1 orders |
| **Potential Loyalists** | $R \in [4,5], F \in [2,3]$ | 178 | **4.1%** | **1.1%** | £531.92 | 23.4 days | 2.0 orders |
| **About to Sleep** | $R \in [2,3], F \in [1,2]$ | 199 | **4.6%** | **1.0%** | £466.94 | 108.9 days | 1.2 orders |
| **Promising** | $R \in [3,4], F = 1$ | 164 | **3.8%** | **0.8%** | £420.28 | 45.2 days | 1.0 orders |
| **New Customers** | $R \in [4,5], F = 1$ | 141 | **3.2%** | **0.6%** | £365.14 | 15.8 days | 1.0 orders |

---

## Machine Learning Clustering (K-Means)

To discover empirical behavioral clusters without rule-based assumptions:
1. **Variance Stabilization:** Natural log transformation $\log(x + 1)$ applied to de-skew power-law distributions.
2. **Standardization:** `StandardScaler` applied to prevent scale dominance.
3. **Hyperparameter Selection:** Evaluating $k \in [2, 8]$ identified an optimal elbow at **$k = 4$** ($\text{Silhouette} = 0.337$).

### Empirical Cluster Profiles:
- **Cluster 1 (High-Value VIPs):** 16.5% of customers producing **64.9% of revenue** (Avg Spend: **£8,074**, Avg Frequency: **13.7**).
- **Cluster 2 (Frequent Regulars):** 27.0% of customers generating **23.7% of revenue** (Avg Spend: **£1,802**, Avg Frequency: **4.1**).
- **Cluster 0 (Recent Shoppers):** 19.3% of customers (Avg Recency: **18 days**, Avg Spend: **£551**).
- **Cluster 3 (Dormant Churn Risks):** 37.2% of customers contributing only 6.2% of revenue (Avg Recency: **182 days**).

---

## Key Insights & Strategic Recommendations

1. **Protect the VIP Core (Champions):** Top buyers account for nearly two-thirds of total turnover. Deploy high-touch VIP relationship programs, dedicated support, and preview product allocations rather than generic discounts.
2. **Tackle the Month-1 Retention Cliff:** Over 79% of customers do not repurchase after month 1. Deploy automated onboarding email drip campaigns within **14 days** of initial order, featuring cross-sell guides and 2nd-purchase shipping incentives.
3. **Automate Win-Back for At-Risk Customers:** 717 customers (£723k past spend) have lapsed over 150+ days. Implement dynamic win-back workflows with personalized re-engagement incentives before transition to the lost pool.

---

## Technologies & Libraries

- **Language:** Python 3.8+
- **Data Wrangling:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`StandardScaler`, `KMeans`, `silhouette_score`)
- **Data Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, JupyterLab

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

### Option A: Run the CLI Automated Pipeline
Execute the complete end-to-end data processing, cohort analysis, RFM segmentation, K-Means clustering, and chart generation in a single command:
```bash
python main.py
```
Outputs generated:
- High-resolution visual charts in `figures/`
- Enriched customer dataset with segment labels in `data/customer_rfm_segments.csv`

### Option B: Interactive Jupyter Notebook
Launch Jupyter to explore interactive markdown narration, step-by-step calculations, and inline visualizations:
```bash
jupyter notebook notebooks/Customer_Segmentation_and_Cohort_Analysis.ipynb
```

---

## Author & Acknowledgments

- **Author:** Jayakumar P
- **GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)
- **Dataset:** [UCI Machine Learning Repository: Online Retail Dataset](https://archive.icsuci.edu/dataset/352/online+retail)
