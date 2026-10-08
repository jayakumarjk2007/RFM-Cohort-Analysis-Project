"""RFM (Recency, Frequency, Monetary) analysis and rule-based customer segmentation."""

import datetime as dt
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_rfm_metrics(df: pd.DataFrame, reference_date: dt.datetime = None) -> pd.DataFrame:
    """Calculate Recency, Frequency, and Monetary metrics for each customer."""
    if reference_date is None:
        # Benchmark reference point: 1 day past the maximum observed transaction
        reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    print(f"[RFM] Computing RFM metrics with reference date: {reference_date.strftime('%Y-%m-%d %H:%M:%S')}")

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (reference_date - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    # Ensure monetary values are non-negative
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)
    return rfm


def assign_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assign 1-5 quantile scores to Recency, Frequency, and Monetary metrics."""
    df = rfm_df.copy()

    # Recency: Lower days = better = higher score (5 is most recent)
    r_labels = [5, 4, 3, 2, 1]
    df["R_Score"] = pd.qcut(df["Recency"], q=5, labels=r_labels).astype(int)

    # Frequency: Higher orders = better = higher score (5 is most frequent)
    # Frequency often has identical low values; use rank method to avoid duplicate bin errors
    f_labels = [1, 2, 3, 4, 5]
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), q=5, labels=f_labels).astype(int)

    # Monetary: Higher spend = better = higher score (5 is highest spending)
    m_labels = [1, 2, 3, 4, 5]
    df["M_Score"] = pd.qcut(df["Monetary"], q=5, labels=m_labels).astype(int)

    # Composite RFM Score strings and values
    df["RFM_Segment"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

    return df


def _classify_rfm(row: pd.Series) -> str:
    """Map R and F scores to standard e-commerce customer segments."""
    r = row["R_Score"]
    f = row["F_Score"]

    if r in [4, 5] and f in [4, 5]:
        return "Champions"
    elif r in [3, 4, 5] and f in [3, 4, 5]:
        return "Loyal Customers"
    elif r in [4, 5] and f in [2, 3]:
        return "Potential Loyalists"
    elif r in [4, 5] and f == 1:
        return "New Customers"
    elif r in [3, 4] and f == 1:
        return "Promising"
    elif r in [2, 3] and f in [2, 3]:
        return "Customers Needing Attention"
    elif r in [2, 3] and f in [1, 2]:
        return "About to Sleep"
    elif r in [1, 2] and f in [2, 3, 4, 5]:
        return "At Risk"
    elif r == 1 and f in [4, 5]:
        return "Can't Lose Them"
    else:
        return "Hibernating / Lost"


def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assign human-readable segment names based on R and F scores."""
    df = rfm_df.copy()
    if "R_Score" not in df.columns:
        df = assign_rfm_scores(df)

    df["Customer_Segment"] = df.apply(_classify_rfm, axis=1)
    return df


def summarize_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Generate aggregate profile summary for each customer segment."""
    total_customers = len(rfm_df)
    total_revenue = rfm_df["Monetary"].sum()

    summary = rfm_df.groupby("Customer_Segment").agg(
        Customer_Count=("CustomerID", "count"),
        Recency_Mean=("Recency", "mean"),
        Frequency_Mean=("Frequency", "mean"),
        Monetary_Mean=("Monetary", "mean"),
        Monetary_Total=("Monetary", "sum"),
    ).reset_index()

    summary["Customer_Share_%"] = (summary["Customer_Count"] / total_customers * 100).round(2)
    summary["Revenue_Share_%"] = (summary["Monetary_Total"] / total_revenue * 100).round(2)
    summary["Recency_Mean"] = summary["Recency_Mean"].round(1)
    summary["Frequency_Mean"] = summary["Frequency_Mean"].round(1)
    summary["Monetary_Mean"] = summary["Monetary_Mean"].round(2)
    summary["Monetary_Total"] = summary["Monetary_Total"].round(2)

    return summary.sort_values(by="Monetary_Total", ascending=False)
