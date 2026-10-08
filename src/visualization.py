"""Visualization utilities for EDA, Cohort Analysis, RFM, and K-Means."""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Clean aesthetic defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def ensure_dir(filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def plot_cohort_retention_heatmap(retention_matrix: pd.DataFrame, save_path: str = "figures/cohort_retention_heatmap.png"):
    """Plot monthly customer retention rate heatmap with percentage annotations."""
    ensure_dir(save_path)
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        retention_matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0.0,
        vmax=50.0,
        cbar_kws={"label": "Retention Rate (%)"},
        linewidths=0.5,
    )
    plt.title("Monthly Cohort Retention Rate (%)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Cohort Period (Months Since First Acquisition)", fontsize=11, fontweight="bold")
    plt.ylabel("Acquisition Cohort Month", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved Cohort Retention Heatmap to {save_path}")


def plot_cohort_retention_curve(retention_matrix: pd.DataFrame, save_path: str = "figures/cohort_retention_curves.png"):
    """Plot retention decay curves for major acquisition cohorts."""
    ensure_dir(save_path)
    plt.figure(figsize=(12, 6))

    # Plot average decay
    mean_retention = retention_matrix.mean(axis=0)
    plt.plot(mean_retention.index, mean_retention.values, marker="o", color="#1f77b4", linewidth=3, label="Overall Average Retention")

    # Plot first 3 cohorts
    for cohort in retention_matrix.index[:4]:
        series = retention_matrix.loc[cohort].dropna()
        plt.plot(series.index, series.values, linestyle="--", alpha=0.6, label=f"Cohort {cohort}")

    plt.title("Cohort Retention Decay Curves Over Customer Lifetime", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Cohort Period (Months)", fontsize=11, fontweight="bold")
    plt.ylabel("Retention Rate (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved Cohort Retention Curves to {save_path}")


def plot_rfm_distributions(rfm_df: pd.DataFrame, save_path: str = "figures/rfm_distributions.png"):
    """Plot histograms of Recency, Frequency, and Monetary before and after log transformation."""
    ensure_dir(save_path)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Raw distributions
    sns.histplot(rfm_df["Recency"], kde=True, ax=axes[0, 0], color="#2b5c8f")
    axes[0, 0].set_title("Recency (Days) - Raw", fontweight="bold")

    sns.histplot(rfm_df["Frequency"].clip(upper=rfm_df["Frequency"].quantile(0.99)), kde=True, ax=axes[0, 1], color="#288059")
    axes[0, 1].set_title("Frequency (Orders) - 99th Pct", fontweight="bold")

    sns.histplot(rfm_df["Monetary"].clip(upper=rfm_df["Monetary"].quantile(0.99)), kde=True, ax=axes[0, 2], color="#c0392b")
    axes[0, 2].set_title("Monetary Spend (£) - 99th Pct", fontweight="bold")

    # Log-transformed distributions
    sns.histplot(np.log1p(rfm_df["Recency"]), kde=True, ax=axes[1, 0], color="#2b5c8f")
    axes[1, 0].set_title("Log(Recency + 1)", fontweight="bold")

    sns.histplot(np.log1p(rfm_df["Frequency"]), kde=True, ax=axes[1, 1], color="#288059")
    axes[1, 1].set_title("Log(Frequency + 1)", fontweight="bold")

    sns.histplot(np.log1p(rfm_df["Monetary"]), kde=True, ax=axes[1, 2], color="#c0392b")
    axes[1, 2].set_title("Log(Monetary + 1)", fontweight="bold")

    plt.suptitle("RFM Metric Distributions: Raw vs. Log-Transformed", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved RFM Distributions to {save_path}")


def plot_rfm_segments(segment_summary: pd.DataFrame, save_path: str = "figures/rfm_segment_summary.png"):
    """Plot horizontal comparative bar charts of Customer Share vs Revenue Share by segment."""
    ensure_dir(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    df_sorted = segment_summary.sort_values(by="Customer_Share_%", ascending=True)

    axes[0].barh(df_sorted["Customer_Segment"], df_sorted["Customer_Share_%"], color="#3498db")
    axes[0].set_title("Customer Share by Segment (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Percentage of Total Customer Base (%)", fontweight="bold")
    for i, v in enumerate(df_sorted["Customer_Share_%"]):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    df_rev = segment_summary.sort_values(by="Customer_Share_%", ascending=True)
    axes[1].barh(df_rev["Customer_Segment"], df_rev["Revenue_Share_%"], color="#2ecc71")
    axes[1].set_title("Revenue Contribution by Segment (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Percentage of Total Revenue (%)", fontweight="bold")
    for i, v in enumerate(df_rev["Revenue_Share_%"]):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.suptitle("RFM Segmentation: Customer Base vs. Revenue Contribution", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved RFM Segment Summary to {save_path}")


def plot_kmeans_elbow_and_silhouette(inertias: List[float], silhouettes: List[float], k_range: range, save_path: str = "figures/kmeans_evaluation.png"):
    """Plot Elbow Inertia and Silhouette Scores across k clusters."""
    ensure_dir(save_path)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = "#2980b9"
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia (Sum of Squared Distances)", color=color, fontweight="bold")
    ax1.plot(list(k_range), inertias, marker="o", color=color, linewidth=2.5)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#e67e22"
    ax2.set_ylabel("Silhouette Score", color=color, fontweight="bold")
    ax2.plot(list(k_range), silhouettes, marker="s", linestyle="--", color=color, linewidth=2.5)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("K-Means Cluster Optimization: Elbow Curve & Silhouette Scores", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved K-Means Evaluation to {save_path}")


def plot_eda_overview(df: pd.DataFrame, save_path: str = "figures/eda_overview.png"):
    """Plot high-level EDA: monthly revenue trend, top 10 products, and country distribution."""
    ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Monthly revenue trend
    monthly_rev = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum() / 1e3
    axes[0].plot(monthly_rev.index.strftime("%b %y"), monthly_rev.values, marker="o", color="#8e44ad", linewidth=2.5)
    axes[0].set_title("Monthly Revenue (£ Thousands)", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Top 10 products by revenue
    top_prods = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10) / 1e3
    axes[1].barh(top_prods.index[::-1], top_prods.values[::-1], color="#16a085")
    axes[1].set_title("Top 10 Products by Revenue (£K)", fontweight="bold")
    axes[1].set_xlabel("Revenue (£ Thousands)")

    # Top 5 countries (excluding UK for scale, with UK note)
    intl_countries = df[df["Country"] != "United Kingdom"]["Country"].value_counts().head(5)
    axes[2].bar(intl_countries.index, intl_countries.values, color="#e74c3c")
    axes[2].set_title("Top 5 International Markets (Excl. UK)", fontweight="bold")
    axes[2].set_ylabel("Transaction Count")
    axes[2].tick_params(axis="x", rotation=30)

    plt.suptitle("Exploratory Data Analysis: Revenue Trends & Geographies", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved EDA Overview to {save_path}")
