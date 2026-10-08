"""Main execution pipeline for Customer Segmentation and Cohort Analysis."""

import argparse
from pathlib import Path
import sys
import time

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cohort_analysis import (
    compute_cohorts,
    generate_cohort_matrix,
    generate_retention_matrix,
    get_cohort_summary_insights,
)
from src.data_loader import clean_data, get_data_summary, load_data
from src.kmeans_segmentation import (
    find_optimal_clusters,
    fit_kmeans,
    preprocess_rfm_for_clustering,
    profile_clusters,
)
from src.rfm_analysis import (
    assign_rfm_scores,
    compute_rfm_metrics,
    segment_customers,
    summarize_segments,
)
from src.visualization import (
    plot_cohort_retention_curve,
    plot_cohort_retention_heatmap,
    plot_eda_overview,
    plot_kmeans_elbow_and_silhouette,
    plot_rfm_distributions,
    plot_rfm_segments,
)


def run_pipeline(data_path: str = None, output_dir: str = None):
    start_time = time.time()
    if data_path is None:
        data_path = str(PROJECT_ROOT / "data" / "online_retail.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "figures")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CUSTOMER SEGMENTATION & COHORT RETENTION ANALYSIS PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Data Loading & Preprocessing
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading and Cleaning Transactions...")
    raw_df = load_data(data_path)
    clean_df, clean_stats = clean_data(raw_df)
    summary = get_data_summary(clean_df)

    print(f"  * Date Span: {summary['date_min']} to {summary['date_max']}")
    print(f"  * Clean Transactions: {summary['total_records']:,}")
    print(f"  * Unique Customers: {summary['unique_customers']:,}")
    print(f"  * Total Revenue: £{summary['total_revenue']:,.2f}")
    print(f"  * Average Order Value: £{summary['average_order_value']:.2f}")

    plot_eda_overview(clean_df, f"{output_dir}/eda_overview.png")

    # -------------------------------------------------------------
    # Step 2: Cohort Analysis & Retention Dynamics
    # -------------------------------------------------------------
    print("\n[Step 2/5] Performing Monthly Cohort Analysis...")
    df_cohort = compute_cohorts(clean_df)
    cohort_counts, cohort_sizes = generate_cohort_matrix(df_cohort)
    retention_matrix = generate_retention_matrix(cohort_counts)
    cohort_insights = get_cohort_summary_insights(retention_matrix)

    print(f"  * Total Acquisition Cohorts Tracked: {cohort_insights['cohorts_count']}")
    print(f"  * Average Month-1 Retention Rate: {cohort_insights['avg_m1_retention']}%")
    if cohort_insights["avg_m3_retention"]:
        print(f"  * Average Month-3 Retention Rate: {cohort_insights['avg_m3_retention']}%")

    plot_cohort_retention_heatmap(retention_matrix, f"{output_dir}/cohort_retention_heatmap.png")
    plot_cohort_retention_curve(retention_matrix, f"{output_dir}/cohort_retention_curves.png")

    # -------------------------------------------------------------
    # Step 3: RFM Metrics & Rule-Based Segmentation
    # -------------------------------------------------------------
    print("\n[Step 3/5] Calculating RFM Metrics & Customer Segments...")
    rfm_df = compute_rfm_metrics(clean_df)
    rfm_scored = assign_rfm_scores(rfm_df)
    rfm_segmented = segment_customers(rfm_scored)
    segment_summary = summarize_segments(rfm_segmented)

    print("\n" + "-" * 75)
    print(f"{'Segment':<28} {'Customers':>10} {'Cust %':>8} {'Revenue %':>10} {'Mean Spend (£)':>15}")
    print("-" * 75)
    for _, row in segment_summary.iterrows():
        print(f"{row['Customer_Segment']:<28} {row['Customer_Count']:>10,} {row['Customer_Share_%']:>7.1f}% {row['Revenue_Share_%']:>9.1f}% {row['Monetary_Mean']:>14.2f}")
    print("-" * 75)

    plot_rfm_distributions(rfm_segmented, f"{output_dir}/rfm_distributions.png")
    plot_rfm_segments(segment_summary, f"{output_dir}/rfm_segment_summary.png")

    # -------------------------------------------------------------
    # Step 4: Machine Learning Clustering (K-Means)
    # -------------------------------------------------------------
    print("\n[Step 4/5] Training K-Means Unsupervised Segmentation...")
    scaled_array, scaler, scaled_df = preprocess_rfm_for_clustering(rfm_segmented)
    inertias, silhouettes = find_optimal_clusters(scaled_array, k_range=range(2, 9))
    plot_kmeans_elbow_and_silhouette(inertias, silhouettes, range(2, 9), f"{output_dir}/kmeans_evaluation.png")

    optimal_k = 4
    km_model, cluster_labels = fit_kmeans(scaled_array, k=optimal_k, random_state=42)
    rfm_segmented["KMeans_Cluster"] = cluster_labels
    cluster_profiles = profile_clusters(rfm_segmented, cluster_labels)

    print(f"\nK-Means Cluster Profiles (k={optimal_k}):")
    print("-" * 80)
    print(f"{'Cluster':<8} {'Archetype':<26} {'Count':>7} {'Cust %':>7} {'Rev %':>7} {'Avg R':>7} {'Avg F':>7} {'Avg M (£)':>11}")
    print("-" * 80)
    for _, row in cluster_profiles.iterrows():
        print(f"C{int(row['Cluster']):<7} {str(row['Archetype']):<26} {row['Count']:>7,} {row['Customer_Share_%']:>6.1f}% {row['Revenue_Share_%']:>6.1f}% {row['Recency_Mean']:>7.1f} {row['Frequency_Mean']:>7.1f} {row['Monetary_Mean']:>10.1f}")
    print("-" * 80)

    # -------------------------------------------------------------
    # Step 5: Export Enriched Dataset
    # -------------------------------------------------------------
    export_path = PROJECT_ROOT / "data" / "customer_rfm_segments.csv"
    rfm_segmented.to_csv(export_path, index=False)
    print(f"\n[Step 5/5] Exported enriched customer segments to {export_path}")

    elapsed = time.time() - start_time
    print(f"\n[Done] Pipeline finished successfully in {elapsed:.1f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RFM and Cohort Analysis Pipeline.")
    parser.add_argument("--data", default=None, help="Path to transactional dataset CSV.")
    parser.add_argument("--outdir", default=None, help="Output directory for generated plots.")
    args = parser.parse_args()

    run_pipeline(data_path=args.data, output_dir=args.outdir)
