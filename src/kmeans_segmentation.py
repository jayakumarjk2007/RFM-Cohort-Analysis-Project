"""K-Means Clustering module for unsupervised customer segmentation."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def preprocess_rfm_for_clustering(rfm_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Transform right-skewed RFM features using natural logarithm and z-score standardization."""
    features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()

    # Log transformation: log(x + 1) to stabilize variance and remove severe right skew
    log_features = np.log1p(features)

    # Standardization: zero mean, unit variance
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(log_features)
    scaled_df = pd.DataFrame(scaled_array, columns=["Recency_Scaled", "Frequency_Scaled", "Monetary_Scaled"])

    return scaled_array, scaler, scaled_df


def find_optimal_clusters(scaled_features: np.ndarray, k_range: range = range(2, 11)) -> Tuple[List[float], List[float]]:
    """Evaluate Inertia (Elbow Method) and Silhouette Score across a range of k values."""
    inertias: List[float] = []
    silhouettes: List[float] = []

    print(f"[KMeans] Evaluating k from {k_range.start} to {k_range.stop - 1}...")
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(scaled_features)
        inertias.append(float(km.inertia_))
        score = float(silhouette_score(scaled_features, labels))
        silhouettes.append(score)
        print(f"  k={k}: Inertia={km.inertia_:.1f}, Silhouette={score:.4f}")

    return inertias, silhouettes


def fit_kmeans(scaled_features: np.ndarray, k: int = 4, random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
    """Train final KMeans model on scaled features."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
    labels = km.fit_predict(scaled_features)
    return km, labels


def profile_clusters(rfm_df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Compute aggregate statistical profiles and business archetypes for each K-Means cluster."""
    df = rfm_df.copy()
    df["Cluster"] = cluster_labels
    total_customers = len(df)
    total_revenue = df["Monetary"].sum()

    profiles = df.groupby("Cluster").agg(
        Count=("CustomerID", "count"),
        Recency_Mean=("Recency", "mean"),
        Frequency_Mean=("Frequency", "mean"),
        Monetary_Mean=("Monetary", "mean"),
        Monetary_Total=("Monetary", "sum"),
    ).reset_index()

    profiles["Customer_Share_%"] = (profiles["Count"] / total_customers * 100).round(2)
    profiles["Revenue_Share_%"] = (profiles["Monetary_Total"] / total_revenue * 100).round(2)
    profiles["Recency_Mean"] = profiles["Recency_Mean"].round(1)
    profiles["Frequency_Mean"] = profiles["Frequency_Mean"].round(1)
    profiles["Monetary_Mean"] = profiles["Monetary_Mean"].round(2)
    profiles["Monetary_Total"] = profiles["Monetary_Total"].round(2)

    # Assign archetype labels based on ranking
    archetypes = {}
    sorted_by_revenue = profiles.sort_values(by="Monetary_Mean", ascending=False)["Cluster"].tolist()
    archetypes[sorted_by_revenue[0]] = "High-Value VIPs (Champions)"
    archetypes[sorted_by_revenue[1]] = "Frequent Regulars"
    if len(sorted_by_revenue) > 2:
        # Check recency of remainder
        rem1 = sorted_by_revenue[2]
        rem2 = sorted_by_revenue[3] if len(sorted_by_revenue) > 3 else None
        profiles.set_index("Cluster", inplace=True)
        if rem2 is not None:
            if profiles.loc[rem1, "Recency_Mean"] < profiles.loc[rem2, "Recency_Mean"]:
                archetypes[rem1] = "Recent Shoppers"
                archetypes[rem2] = "Dormant / Churn Risks"
            else:
                archetypes[rem1] = "Dormant / Churn Risks"
                archetypes[rem2] = "Recent Shoppers"
        else:
            archetypes[rem1] = "Occasional / Low Value"
        profiles.reset_index(inplace=True)

    profiles["Archetype"] = profiles["Cluster"].map(archetypes)
    return profiles.sort_values(by="Monetary_Total", ascending=False)
