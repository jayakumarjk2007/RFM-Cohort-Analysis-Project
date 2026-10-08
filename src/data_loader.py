"""Data loading, inspection, and cleaning module for Online Retail dataset."""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from CSV or Excel file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    print(f"[DataLoader] Loading data from {path.name}...")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"[DataLoader] Successfully loaded {len(df):,} raw records with {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean the raw transactions by handling missing values, cancellations, and invalid records."""
    initial_rows = len(df)
    stats: Dict[str, int] = {"initial_rows": initial_rows}

    # 1. Drop missing CustomerID
    missing_customer = df["CustomerID"].isna().sum()
    stats["missing_customer_ids"] = int(missing_customer)
    df_clean = df.dropna(subset=["CustomerID"]).copy()
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

    # 2. Filter out canceled orders (Invoice numbers starting with 'C')
    canceled_orders = df_clean["InvoiceNo"].astype(str).str.startswith("C").sum()
    stats["canceled_orders"] = int(canceled_orders)
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]

    # 3. Filter valid Quantity and UnitPrice
    invalid_rows = ((df_clean["Quantity"] <= 0) | (df_clean["UnitPrice"] <= 0)).sum()
    stats["invalid_quantity_price"] = int(invalid_rows)
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

    # 4. Standardize datetime and compute TotalPrice
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    # 5. Clean text fields
    if "Description" in df_clean.columns:
        df_clean["Description"] = df_clean["Description"].astype(str).str.strip()

    final_rows = len(df_clean)
    stats["final_rows"] = final_rows
    stats["removed_rows"] = initial_rows - final_rows
    stats["unique_customers"] = df_clean["CustomerID"].nunique()
    stats["unique_products"] = df_clean["StockCode"].nunique()

    print(f"[DataLoader] Cleaning summary: Removed {stats['removed_rows']:,} records ({stats['removed_rows']/initial_rows*100:.1f}%).")
    print(f"[DataLoader] Clean dataset contains {final_rows:,} records across {stats['unique_customers']:,} unique customers.")
    return df_clean, stats


def get_data_summary(df: pd.DataFrame) -> Dict[str, any]:
    """Calculate descriptive statistics for the cleaned dataset."""
    return {
        "total_records": len(df),
        "unique_customers": df["CustomerID"].nunique(),
        "unique_invoices": df["InvoiceNo"].nunique(),
        "unique_products": df["StockCode"].nunique(),
        "date_min": df["InvoiceDate"].min().strftime("%Y-%m-%d"),
        "date_max": df["InvoiceDate"].max().strftime("%Y-%m-%d"),
        "total_revenue": round(float(df["TotalPrice"].sum()), 2),
        "total_units_sold": int(df["Quantity"].sum()),
        "average_order_value": round(float(df.groupby("InvoiceNo")["TotalPrice"].sum().mean()), 2),
        "top_countries": df["Country"].value_counts().head(5).to_dict(),
    }
