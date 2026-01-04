import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class GeometricAnalyzer:
    """Analyze geometric properties of data using different distance metrics"""

    def __init__(self, data, feature_subset=None):
        self.data = data.copy()
        self.feature_names = data.columns.tolist()

        if feature_subset:
            self.feature_subset = [f for f in feature_subset if f in self.feature_names]
            if len(self.feature_subset) == 0:
                self.feature_subset = self.feature_names[:min(10, len(self.feature_names))]
        else:
            self.feature_subset = self.feature_names[:min(10, len(self.feature_names))]

        self.X = self.data[self.feature_subset].values
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.results = {}

    def compute_distance_matrices(self, metrics=['euclidean', 'cosine', 'cityblock', 'correlation'],
                                  subsample=None, random_state=42):
        """Compute distance matrices using different metrics"""

        # Subsample if requested
        if subsample and subsample < len(self.X_scaled):
            np.random.seed(random_state)
            idx = np.random.choice(len(self.X_scaled), subsample, replace=False)
            X_sampled = self.X_scaled[idx]
            sample_indices = idx
        else:
            X_sampled = self.X_scaled
            sample_indices = np.arange(len(self.X_scaled))

        distance_matrices = []
        metric_properties = []

        for metric in metrics:
            if metric == 'euclidean':
                D = euclidean_distances(X_sampled)
            elif metric == 'cosine':
                D = cosine_distances(X_sampled)
            elif metric == 'cityblock' or metric == 'manhattan':
                D = manhattan_distances(X_sampled)
            elif metric == 'correlation':
                D = cdist(X_sampled, X_sampled, metric='correlation')
            else:
                continue

            distance_matrices.append(D)

            properties = {
                'metric': metric,
                'mean_distance': np.mean(D),
                'std_distance': np.std(D),
                'min_distance': np.min(D),
                'max_distance': np.max(D)
            }
            metric_properties.append(properties)

        self.results = {
            'distance_matrices': distance_matrices,
            'metrics': [m for m in metrics if m in ['euclidean', 'cosine', 'cityblock', 'correlation']],
            'metric_properties': pd.DataFrame(metric_properties),
            'sample_indices': sample_indices,
            'X_sampled': X_sampled
        }

        return self.results

    def compare_nearest_neighbors(self, k=10):
        """Compare nearest neighbors across different metrics"""
        if 'distance_matrices' not in self.results:
            return None

        metrics = self.results['metrics']
        distance_matrices = self.results['distance_matrices']
        n_samples = len(self.results['X_sampled'])

        # Get neighbor sets for each metric
        neighbor_sets = []

        for D in distance_matrices:
            neighbors = []
            for j in range(n_samples):
                dists = D[j]
                dists_without_self = dists.copy()
                dists_without_self[j] = np.inf
                nearest_idx = np.argsort(dists_without_self)[:k]
                neighbors.append(set(nearest_idx))
            neighbor_sets.append(neighbors)

        # Compute agreement matrix
        n_metrics = len(metrics)
        agreement_matrix = np.zeros((n_metrics, n_metrics))

        for i in range(n_metrics):
            for j in range(n_metrics):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreements = []
                    for sample_idx in range(n_samples):
                        set_i = neighbor_sets[i][sample_idx]
                        set_j = neighbor_sets[j][sample_idx]
                        intersection = len(set_i.intersection(set_j))
                        union = len(set_i.union(set_j))
                        jaccard = intersection / union if union > 0 else 0
                        agreements.append(jaccard)

                    agreement_matrix[i, j] = np.mean(agreements)

        agreement_df = pd.DataFrame(
            agreement_matrix,
            index=metrics,
            columns=metrics
        )

        return agreement_df.round(3)

    def visualize_analysis(self, save_path=None):
        """Create visualization of distance geometry analysis"""
        # Simplified visualization - you can expand this
        import matplotlib.pyplot as plt

        if 'distance_matrices' not in self.results:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot distance distributions
        metrics = self.results['metrics']
        distance_matrices = self.results['distance_matrices']
        colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))

        for i, (metric, D, color) in enumerate(zip(metrics, distance_matrices, colors)):
            n = len(D)
            dist_vals = D[np.triu_indices(n, k=1)]
            axes[0].hist(dist_vals, bins=50, alpha=0.5, label=metric,
                        density=True, color=color, edgecolor='black', linewidth=0.5)

        axes[0].set_xlabel('Distance')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distance Distributions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot agreement matrix
        agreement = self.compare_nearest_neighbors(k=5)
        if agreement is not None:
            im = axes[1].imshow(agreement.values, cmap='YlOrRd', vmin=0, vmax=1)
            axes[1].set_xticks(np.arange(len(metrics)))
            axes[1].set_yticks(np.arange(len(metrics)))
            axes[1].set_xticklabels(metrics, rotation=45, ha='right')
            axes[1].set_yticklabels(metrics)
            axes[1].set_title('Neighbor Agreement (k=5)')
            plt.colorbar(im, ax=axes[1], label='Jaccard Similarity')

        plt.suptitle('Distance Geometry Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def generate_report(self):
        """Generate analysis report"""
        if 'distance_matrices' not in self.results:
            return None

        properties = self.results['metric_properties']

        # Determine best metrics
        best_clustering_idx = properties['std_distance'].idxmax()
        best_clustering_metric = properties.loc[best_clustering_idx, 'metric']

        properties['compactness_score'] = properties['mean_distance'] / (properties['std_distance'] + 1e-6)
        best_nn_idx = properties['compactness_score'].idxmin()
        best_nn_metric = properties.loc[best_nn_idx, 'metric']

        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(self.data),
            'n_features': len(self.feature_subset),
            'feature_names': self.feature_subset,
            'metrics_tested': self.results['metrics'],
            'best_clustering_metric': best_clustering_metric,
            'best_nn_metric': best_nn_metric
        }

        return report