"""Small visualization helpers used by the project.

This file intentionally provides minimal, well-behaved functions so
notebooks and modules can import `src.visualizations` without errors.
Add more helpers here as needed.
"""
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd


def save_fig(fig, path: str, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches='tight')


def plot_neighbor_overlap(matrix: pd.DataFrame, ax=None):
    import seaborn as sns
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Neighbor Overlap (k)')
    return ax
