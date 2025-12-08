#!/usr/bin/env python3
"""
Spatial Graph Construction for ST-GNN

This module builds the spatial graph structure connecting households based on:
- Geographic proximity (province/municipality)
- K-nearest neighbors in feature space
- Contracted power similarity

The graph is consistent across all splits (train/val/test/CL).

Author: Energy-Efficient STGNN Project
Date: December 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Project root (…/energy_efficient_stgnn)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use Path objects anchored at project root
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "7362094" / "metadata.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Graph construction parameters
K_NEIGHBORS = 10  # Number of nearest neighbors
INCLUDE_SELF_LOOPS = True
NORMALIZE_ADJACENCY = True

# Feature weights for similarity computation
FEATURE_WEIGHTS = {
    'geographic': 0.5,  # Province/municipality similarity
    'power': 0.3,  # Contracted power (P1-P6)
    'tariff': 0.2  # Tariff type similarity
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load household metadata from CSV.

    Args:
        metadata_path (Path): Path to metadata.csv

    Returns:
        pd.DataFrame: Metadata for all households
    """
    print(f"Loading metadata from: {metadata_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    print(f"  ✓ Loaded {len(df):,} household records")
    print(f"  ✓ Columns: {list(df.columns)}")

    return df


def get_train_households(splits_dir: Path):
    """
    Get list of household IDs present in the training split.

    Args:
        splits_dir (Path): Path to train/ directory

    Returns:
        list: Household IDs (without .csv extension)
    """
    print(f"\nIdentifying households in training split...")

    if not splits_dir.exists():
        raise FileNotFoundError(f"Training split directory not found: {splits_dir}")

    csv_files = list(splits_dir.glob("*.csv"))
    household_ids = [f.stem for f in csv_files]

    print(f"  ✓ Found {len(household_ids):,} households in training split")

    return sorted(household_ids)


def filter_metadata(metadata_df, household_ids):
    """
    Filter metadata to only include households in training split.

    Args:
        metadata_df (pd.DataFrame): Full metadata
        household_ids (list): List of household IDs to keep

    Returns:
        pd.DataFrame: Filtered metadata
    """
    print(f"\nFiltering metadata to match training households...")

    # Filter by user hash
    filtered_df = metadata_df[metadata_df['user'].isin(household_ids)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    print(f"  ✓ Kept {len(filtered_df):,} households")

    # Verify all households are present
    missing = set(household_ids) - set(filtered_df['user'])
    if missing:
        print(f"  ⚠ Warning: {len(missing)} households not found in metadata")
        print(f"    First 5 missing: {list(missing)[:5]}")

    return filtered_df


def create_node_mapping(household_ids):
    """
    Create mapping from household ID to node index.

    Args:
        household_ids (list): Sorted list of household IDs

    Returns:
        dict: {household_id: node_index}
    """
    print(f"\nCreating node index mapping...")

    node_map = {hh_id: idx for idx, hh_id in enumerate(household_ids)}

    print(f"  ✓ Created mapping for {len(node_map):,} nodes")
    print(f"  ✓ Node indices: 0 to {len(node_map) - 1}")

    return node_map


def extract_features(metadata_df):
    """
    Extract features for similarity computation.

    Args:
        metadata_df (pd.DataFrame): Filtered metadata

    Returns:
        tuple: (geographic_features, power_features, tariff_features)
    """
    print(f"\nExtracting features for graph construction...")

    num_households = len(metadata_df)

    # =====================================================================
    # 1. GEOGRAPHIC FEATURES
    # =====================================================================
    # Encode province and municipality as one-hot vectors

    provinces = metadata_df['province'].fillna('unknown')
    municipalities = metadata_df['municipality'].fillna('unknown')

    # Create combined geographic feature
    province_dummies = pd.get_dummies(provinces, prefix='prov')
    municipality_dummies = pd.get_dummies(municipalities, prefix='muni')

    geographic_features = pd.concat([province_dummies, municipality_dummies], axis=1).values

    print(f"  ✓ Geographic features: {geographic_features.shape[1]} dimensions")

    # =====================================================================
    # 2. CONTRACTED POWER FEATURES
    # =====================================================================
    # Use P1-P6 power limits

    power_cols = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
    power_features = metadata_df[power_cols].fillna(0).values

    # Normalize power features
    scaler = StandardScaler()
    power_features = scaler.fit_transform(power_features)

    print(f"  ✓ Power features: {power_features.shape[1]} dimensions")

    # =====================================================================
    # 3. TARIFF FEATURES
    # =====================================================================
    # One-hot encode contracted tariff

    tariffs = metadata_df['contracted_tariff'].fillna('unknown')
    tariff_features = pd.get_dummies(tariffs, prefix='tariff').values

    print(f"  ✓ Tariff features: {tariff_features.shape[1]} dimensions")

    return geographic_features, power_features, tariff_features


def compute_similarity_matrix(geographic_feat, power_feat, tariff_feat, weights):
    """
    Compute weighted similarity matrix between all households.

    Args:
        geographic_feat (np.ndarray): Geographic features [N, d_geo]
        power_feat (np.ndarray): Power features [N, d_power]
        tariff_feat (np.ndarray): Tariff features [N, d_tariff]
        weights (dict): Feature weights

    Returns:
        np.ndarray: Similarity matrix [N, N] (higher = more similar)
    """
    print(f"\nComputing similarity matrix...")

    num_households = geographic_feat.shape[0]

    # Compute pairwise distances for each feature type
    geo_dist = cdist(geographic_feat, geographic_feat, metric='cosine')
    power_dist = cdist(power_feat, power_feat, metric='euclidean')
    tariff_dist = cdist(tariff_feat, tariff_feat, metric='cosine')

    # Convert distances to similarities (1 - distance)
    geo_sim = 1 - geo_dist
    power_sim = 1 - (power_dist / (power_dist.max() + 1e-8))  # Normalize to [0, 1]
    tariff_sim = 1 - tariff_dist

    # Weighted combination
    similarity = (
        weights['geographic'] * geo_sim +
        weights['power'] * power_sim +
        weights['tariff'] * tariff_sim
    )

    print(f"  ✓ Similarity matrix shape: {similarity.shape}")
    print(f"  ✓ Similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")

    return similarity


def build_knn_adjacency(similarity_matrix, k, include_self_loops=True):
    """
    Build K-nearest neighbor adjacency matrix.

    Args:
        similarity_matrix (np.ndarray): Similarity matrix [N, N]
        k (int): Number of nearest neighbors
        include_self_loops (bool): Whether to include self-connections

    Returns:
        csr_matrix: Sparse adjacency matrix [N, N]
    """
    print(f"\nBuilding K-NN adjacency matrix (k={k})...")

    num_households = similarity_matrix.shape[0]

    # For each node, find k nearest neighbors
    row_indices = []
    col_indices = []
    edge_weights = []

    for i in range(num_households):
        # Get similarities for node i
        similarities = similarity_matrix[i]

        # Don't include self in neighbors (we'll add self-loops separately if needed)
        similarities[i] = -np.inf

        # Find k nearest neighbors
        k_nearest_indices = np.argsort(similarities)[-k:]
        k_nearest_weights = similarities[k_nearest_indices]

        # Add edges
        for j, weight in zip(k_nearest_indices, k_nearest_weights):
            row_indices.append(i)
            col_indices.append(j)
            edge_weights.append(weight)

    # Create sparse adjacency matrix
    adjacency = csr_matrix(
        (edge_weights, (row_indices, col_indices)),
        shape=(num_households, num_households)
    )

    # Make adjacency symmetric (if i connects to j, j connects to i)
    adjacency = adjacency + adjacency.T
    adjacency.data = np.clip(adjacency.data, 0, 1)  # Cap at 1

    # Add self-loops if requested
    if include_self_loops:
        identity = csr_matrix(np.eye(num_households))
        adjacency = adjacency + identity

    num_edges = adjacency.nnz
    avg_degree = num_edges / num_households

    print(f"  ✓ Adjacency matrix shape: {adjacency.shape}")
    print(f"  ✓ Number of edges: {num_edges:,}")
    print(f"  ✓ Average degree: {avg_degree:.2f}")
    print(f"  ✓ Sparsity: {1 - (num_edges / (num_households ** 2)):.4f}")

    return adjacency


def normalize_adjacency(adjacency):
    """
    Normalize adjacency matrix using symmetric normalization.
    D^(-1/2) * A * D^(-1/2)

    Args:
        adjacency (csr_matrix): Adjacency matrix

    Returns:
        csr_matrix: Normalized adjacency matrix
    """
    print(f"\nNormalizing adjacency matrix...")

    # Compute degree matrix
    degrees = np.array(adjacency.sum(axis=1)).flatten()

    # Compute D^(-1/2)
    degrees_inv_sqrt = np.power(degrees, -0.5)
    degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0.0

    # Create diagonal matrix
    D_inv_sqrt = csr_matrix(np.diag(degrees_inv_sqrt))

    # Normalize: D^(-1/2) * A * D^(-1/2)
    adjacency_normalized = D_inv_sqrt @ adjacency @ D_inv_sqrt

    print(f"  ✓ Normalized adjacency matrix")

    return adjacency_normalized


def save_graph(adjacency: csr_matrix, node_map: dict, output_dir: Path):
    """
    Save adjacency matrix and node mapping to disk.

    Args:
        adjacency (csr_matrix): Adjacency matrix
        node_map (dict): Node index mapping
        output_dir (Path): Output directory
    """
    print(f"\nSaving graph to disk...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save adjacency matrix
    adjacency_path = output_dir / "adjacency_matrix.npz"
    save_npz(adjacency_path, adjacency)
    print(f"  ✓ Saved adjacency matrix: {adjacency_path}")

    # Save node mapping
    node_map_path = output_dir / "node_map.json"
    with node_map_path.open('w') as f:
        json.dump(node_map, f, indent=2)
    print(f"  ✓ Saved node mapping: {node_map_path}")

    # Save graph statistics
    stats = {
        'num_nodes': adjacency.shape[0],
        'num_edges': adjacency.nnz,
        'avg_degree': adjacency.nnz / adjacency.shape[0],
        'sparsity': 1 - (adjacency.nnz / (adjacency.shape[0] ** 2))
    }

    stats_path = output_dir / "graph_stats.json"
    with stats_path.open('w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved graph statistics: {stats_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def build_graph():
    """Main graph construction pipeline."""
    print("\n" + "=" * 70)
    print("  SPATIAL GRAPH CONSTRUCTION")
    print("=" * 70)

    try:
        # Step 1: Load metadata
        metadata_df = load_metadata(METADATA_PATH)

        # Step 2: Get training households
        household_ids = get_train_households(SPLITS_DIR)

        # Step 3: Filter metadata to match training households
        filtered_metadata = filter_metadata(metadata_df, household_ids)

        # Step 4: Create node index mapping
        node_map = create_node_mapping(household_ids)

        # Step 5: Extract features
        geo_feat, power_feat, tariff_feat = extract_features(filtered_metadata)

        # Step 6: Compute similarity matrix
        similarity = compute_similarity_matrix(
            geo_feat, power_feat, tariff_feat, FEATURE_WEIGHTS
        )

        # Step 7: Build K-NN adjacency matrix
        adjacency = build_knn_adjacency(
            similarity, K_NEIGHBORS, INCLUDE_SELF_LOOPS
        )

        # Step 8: Normalize adjacency matrix
        if NORMALIZE_ADJACENCY:
            adjacency = normalize_adjacency(adjacency)

        # Step 9: Save graph to disk
        save_graph(adjacency, node_map, OUTPUT_DIR)

        print("\n" + "=" * 70)
        print("✓ GRAPH CONSTRUCTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nOutputs saved to: {OUTPUT_DIR}/")
        print(f"  - adjacency_matrix.npz")
        print(f"  - node_map.json")
        print(f"  - graph_stats.json")

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'=' * 70}")
        raise


if __name__ == "__main__":
    build_graph()
