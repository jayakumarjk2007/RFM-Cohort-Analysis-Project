"""Cohort Analysis module for customer retention and behavioral tracking."""

from typing import Tuple
import pandas as pd


def get_month(date_series: pd.Series) -> pd.Series:
    """Truncate timestamp to the first day of the calendar month."""
    return date_series.dt.to_period("M").dt.to_timestamp()


def compute_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate acquisition CohortMonth, transaction InvoiceMonth, and CohortIndex for each row."""
    df_cohort = df.copy()

    # 1. Truncate InvoiceDate to month
    df_cohort["InvoiceMonth"] = get_month(df_cohort["InvoiceDate"])

    # 2. Identify CohortMonth (the month of customer's first purchase)
    cohort_month = df_cohort.groupby("CustomerID")["InvoiceMonth"].transform("min")
    df_cohort["CohortMonth"] = cohort_month

    # 3. Calculate year and month components
    invoice_year = df_cohort["InvoiceMonth"].dt.year
    invoice_month = df_cohort["InvoiceMonth"].dt.month
    cohort_year = df_cohort["CohortMonth"].dt.year
    cohort_month = df_cohort["CohortMonth"].dt.month

    # 4. CohortIndex (1 = acquisition month, 2 = month 1, ...)
    years_diff = invoice_year - cohort_year
    months_diff = invoice_month - cohort_month
    df_cohort["CohortIndex"] = years_diff * 12 + months_diff + 1

    return df_cohort


def generate_cohort_matrix(df_cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate cohort customer counts matrix and cohort initial size vector."""
    cohort_data = df_cohort.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().reset_index()
    cohort_counts = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID")
    cohort_sizes = cohort_counts.iloc[:, 0]
    return cohort_counts, cohort_sizes


def generate_retention_matrix(cohort_counts: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage retention rate matrix from cohort counts."""
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention_matrix = cohort_counts.divide(cohort_sizes, axis=0) * 100.0
    retention_matrix.index = retention_matrix.index.strftime("%Y-%m")
    return retention_matrix


def generate_average_spend_matrix(df_cohort: pd.DataFrame) -> pd.DataFrame:
    """Calculate average customer spend matrix per cohort over time."""
    cohort_spend = df_cohort.groupby(["CohortMonth", "CohortIndex"])["TotalPrice"].mean().reset_index()
    spend_matrix = cohort_spend.pivot(index="CohortMonth", columns="CohortIndex", values="TotalPrice")
    spend_matrix.index = spend_matrix.index.strftime("%Y-%m")
    return spend_matrix


def get_cohort_summary_insights(retention_matrix: pd.DataFrame) -> dict:
    """Extract key retention decay metrics."""
    m1_retention = retention_matrix[2].dropna()  # Index 2 = Month 1 after acquisition
    m3_retention = retention_matrix[4].dropna() if 4 in retention_matrix.columns else None

    return {
        "avg_m1_retention": round(float(m1_retention.mean()), 2),
        "min_m1_retention": round(float(m1_retention.min()), 2),
        "max_m1_retention": round(float(m1_retention.max()), 2),
        "avg_m3_retention": round(float(m3_retention.mean()), 2) if m3_retention is not None else None,
        "cohorts_count": len(retention_matrix),
    }
