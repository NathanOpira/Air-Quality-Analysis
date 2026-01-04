"""
distance_analysis.py
Core module for geometric analysis of tabular data.
Implements the technical spine: Structure through distance metrics.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class GeometricAnalyzer:
    """Analyzer for distance geometry in tabular data."""

    def __init__(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Initialize with data and features to analyze.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame with datetime index
        feature_columns : List[str]
            Columns to use for distance computations
        """
        self.df = df
        self.features = df[feature_columns].copy()
        self.X = self.features.values
        self.results = {}

    def compute_metric_comparison(self,
                                 metrics: List[str] = ['euclidean', 'cosine', 'cityblock'],
                                 subsample: int = 500) -> Dict:
        """
        Compare how different metrics perceive data geometry.

        Returns
        -------
        Dict with:
        - distance_matrices: 3D array of shape (n_metrics, n_samples, n_samples)
        - neighbor_agreement: Agreement matrix between metrics
        - metric_properties: Summary of each metric's behavior
        """
        print("Computing distance geometry...")

        n_samples = min(subsample, len(self.X))
        indices = np.random.choice(len(self.X), n_samples, replace=False)
        X_sample = self.X[indices]

        # Store all distance matrices
        distance_matrices = []
        metric_properties = []

        for metric in metrics:
            # Compute pairwise distances
            if metric == 'cosine':
                # Normalize for cosine
                norms = np.linalg.norm(X_sample, axis=1, keepdims=True)
                X_norm = X_sample / np.where(norms == 0, 1, norms)
                dist = pdist(X_norm, metric='cosine')
            else:
                dist = pdist(X_sample, metric=metric)

            dist_matrix = squareform(dist)
            distance_matrices.append(dist_matrix)

            # Compute metric properties
            properties = {
                'metric': metric,
                'mean_distance': np.mean(dist),
                'std_distance': np.std(dist),
                'concentration_ratio': np.std(dist) / (np.mean(dist) + 1e-10),
                'rank_stability': self._compute_rank_stability(dist_matrix)
            }
            metric_properties.append(properties)

        # Convert to numpy array
        distance_matrices = np.array(distance_matrices)

        # Compute agreement between metrics
        neighbor_agreement = self._compute_neighbor_agreement(distance_matrices, metrics, k=10)

        self.results['distance_matrices'] = distance_matrices
        self.results['neighbor_agreement'] = neighbor_agreement
        self.results['metric_properties'] = pd.DataFrame(metric_properties)
        self.results['sample_indices'] = indices

        print("Geometry computed.")
        return self.results

    def _compute_rank_stability(self, dist_matrix: np.ndarray) -> float:
        """Compute how stable neighbor rankings are."""
        n = len(dist_matrix)
        ranks = np.argsort(dist_matrix, axis=1)

        # Compare first half vs second half rankings
        split = n // 2
        sample_pairs = min(100, split)

        correlations = []
        for _ in range(sample_pairs):
            i = np.random.randint(split)
            rank1 = ranks[i]
            rank2 = ranks[i + split]
            corr, _ = spearmanr(rank1[:50], rank2[:50])  # Top 50 neighbors
            if not np.isnan(corr):
                correlations.append(corr)

        return np.mean(correlations) if correlations else 0

    def _compute_neighbor_agreement(self,
                                   dist_matrices: np.ndarray,
                                   metrics: List[str],
                                   k: int = 10) -> pd.DataFrame:
        """Compute agreement in k-nearest neighbors between metrics."""
        n_metrics = len(metrics)
        agreement_matrix = np.zeros((n_metrics, n_metrics))

        for i in range(n_metrics):
            for j in range(n_metrics):
                # Get nearest neighbors for each point
                neighbors_i = np.argsort(dist_matrices[i], axis=1)[:, 1:k+1]
                neighbors_j = np.argsort(dist_matrices[j], axis=1)[:, 1:k+1]

                # Compute overlap
                overlaps = []
                for idx in range(len(neighbors_i)):
                    overlap = len(set(neighbors_i[idx]) & set(neighbors_j[idx])) / k
                    overlaps.append(overlap)

                agreement_matrix[i, j] = np.mean(overlaps)

        return pd.DataFrame(agreement_matrix, index=metrics, columns=metrics)

    def visualize_metric_disagreement(self, save_path: str = None):
        """Create publication-quality visualization of metric disagreement."""
        if not self.results:
            raise ValueError("Run compute_metric_comparison() first.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Agreement heatmap
        ax = axes[0, 0]
        sns.heatmap(self.results['neighbor_agreement'],
                   annot=True, fmt='.2f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Neighbor Overlap'})
        ax.set_title('Metric Agreement: k=10 Nearest Neighbors', fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance Metric')
        ax.set_ylabel('Distance Metric')

        # 2. Distance distributions
        ax = axes[0, 1]
        dist_matrices = self.results['distance_matrices']
        metrics = self.results['neighbor_agreement'].index.tolist()

        for i, metric in enumerate(metrics):
            # Flatten upper triangle
            dist_vals = dist_matrices[i][np.triu_indices_from(dist_matrices[i], k=1)]
            sns.kdeplot(dist_vals[:5000], label=metric, ax=ax, linewidth=2)

        ax.set_xlabel('Pairwise Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distance Distribution by Metric', fontsize=14, fontweight='bold')
        ax.legend()

        # 3. Concentration ratio comparison
        ax = axes[1, 0]
        props = self.results['metric_properties']
        colors = plt.cm.Set3(np.linspace(0, 1, len(props)))

        bars = ax.bar(range(len(props)), props['concentration_ratio'],
                     color=colors, edgecolor='black')
        ax.set_xticks(range(len(props)))
        ax.set_xticklabels(props['metric'], rotation=45, ha='right')
        ax.set_ylabel('Distance Concentration (σ/μ)', fontsize=12)
        ax.set_title('Metric Sensitivity to High Dimensions', fontsize=14, fontweight='bold')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. Case study: One point's varying neighbors
        ax = axes[1, 1]
        case_point = 42  # Example point
        dist_matrices = self.results['distance_matrices']

        neighbor_data = []
        for i, metric in enumerate(metrics):
            distances = dist_matrices[i][case_point]
            nearest = np.argsort(distances)[1:11]  # Top 10 excluding self
            for rank, idx in enumerate(nearest):
                neighbor_data.append({
                    'metric': metric,
                    'neighbor_rank': rank + 1,
                    'neighbor_id': idx,
                    'distance': distances[idx]
                })

        neighbor_df = pd.DataFrame(neighbor_data)

        # Pivot for visualization
        pivot_df = neighbor_df.pivot_table(index='neighbor_rank',
                                          columns='metric',
                                          values='neighbor_id',
                                          aggfunc='first')

        for metric in metrics:
            ax.scatter(pivot_df.index, pivot_df[metric],
                      label=metric, s=100, alpha=0.7)

        ax.set_xlabel('Neighbor Rank (1=closest)', fontsize=12)
        ax.set_ylabel('Neighbor ID', fontsize=12)
        ax.set_title(f'Case Study: Neighbors of Point {case_point}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            from pathlib import Path
            outp = Path(save_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()
        return fig

    def generate_technical_report(self, save_path: str = "outputs/geometric_analysis_report.md"):
        """Generate a Markdown report of findings."""
        if not self.results:
            raise ValueError("Run compute_metric_comparison() first.")

        report = [
            "# Geometric Analysis Report: Distance Metric Comparison",
            "",
            "## Executive Summary",
            f"This analysis compares {len(self.results['metric_properties'])} distance metrics ",
            "on their perception of data geometry in air quality data.",
            "",
            "## Key Findings",
            ""
        ]

        # Add metric properties table
        props = self.results['metric_properties']
        report.append("### Metric Properties")
        try:
            report.append(props.to_markdown(index=False))
        except Exception:
            # Fallback if optional dependencies (e.g., tabulate) are missing
            report.append(props.to_string(index=False))
        report.append("")

        # Add interpretation
        report.append("### Interpretation")
        report.append("1. **Concentration Ratio**: Lower values indicate better discrimination in high dimensions.")
        report.append("2. **Rank Stability**: Higher values indicate more consistent neighbor rankings.")
        report.append("")

        # Add agreement matrix
        report.append("### Neighbor Agreement Matrix (k=10)")
        try:
            report.append(self.results['neighbor_agreement'].to_markdown())
        except Exception:
            report.append(self.results['neighbor_agreement'].to_string())
        report.append("")

        report.append("## Recommendations")

        best_metric = props.loc[props['concentration_ratio'].idxmin(), 'metric']
        report.append(f"1. **For clustering**: Use **{best_metric}** (lowest concentration ratio).")

        stable_metric = props.loc[props['rank_stability'].idxmax(), 'metric']
        report.append(f"2. **For nearest neighbor search**: Use **{stable_metric}** (highest rank stability).")

        report.append("3. **Always validate** metric choice with downstream task performance.")

        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)

        print(f"Technical report saved to {save_path}")
        return report_text

def load_and_prepare_data():
    """Helper function to load your air quality data."""
    # ADAPT THIS TO YOUR ACTUAL DATA LOADING
    from src.data_loader import load_processed_data
    from src.feature_engineering import create_analytical_features

    df = load_processed_data()
    X, features = create_analytical_features(df)

    # Select key pollution features for analysis
    pollution_features = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)',
        'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
        'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]

    # Use only features that exist in your data
    available_features = [f for f in pollution_features if f in df.columns]

    return df, available_features