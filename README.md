# The Geometry of Air Quality: A Distance-Based Analysis

## 🎯 Technical Spine
**"Structure, similarity, and behavior in tabular data through distance metrics."**

This project moves beyond traditional air quality prediction to analyze the **geometric structure** of multivariate pollution data. We investigate how different mathematical definitions of "distance" fundamentally change our understanding of similar atmospheric states.

## 🔬 Core Research Questions
1. How do distance metrics (Euclidean, Cosine, Manhattan, Correlation) perceive the same air quality data differently?
2. What geometric properties emerge when we view pollutants as points in high-dimensional space?
3. How does feature engineering affect the data manifold structure?
4. Which metric is optimal for specific tasks (clustering, anomaly detection, forecasting)?

## 📊 Key Findings
### 1. Metric Dependence of Similarity
- **Euclidean distance** groups days by *absolute pollution magnitude*
- **Cosine distance** groups days by *pollutant proportion* (reveals combustion regimes)
- **Correlation distance** ignores magnitude, focuses on *pattern similarity*
- Neighbor agreement between metrics can be as low as 40% (majority disagreement)

### 2. Geometric Regimes Discovered
Using cosine distance on normalized pollutant ratios revealed three atmospheric regimes:
- **Regime A**: High CO/NOx ratio (traffic-dominated)
- **Regime B**: Balanced ratios (background pollution)
- **Regime C**: Low CO/NOx ratio (industrial/atypical)

### 3. Temporal Structure
Cyclic encoding of time (sin/cos transformations) preserves circular geometry, making 23:59 and 00:01 geometrically close as they should be.

## 🛠️ Technical Implementation

### Repository Structure

Air-Quality-Analysis/
├── data/ # Version-controlled data
│ ├── raw/ # Original UCI dataset (immutable)
│ └── processed/ # Cleaned data and feature matrices
├── notebooks/ # Narrative analysis pipeline
│ ├── 01_data_exploration.ipynb
│ ├── 02_cleaning_challenges.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_modeling.ipynb
│ └── 05_distance_geometry.ipynb # Technical spine showcase
├── src/ # Production-grade Python modules
│ ├── data_loader.py # Data loading and preprocessing
│ ├── features.py # Geometric feature engineering
│ ├── distance_analyzer.py # Core distance analysis (GeometricAnalyzer)
│ ├── model_trainer.py # ML model training
│ └── init.py # Package structure
├── outputs/ # Generated artifacts
│ ├── figures/ # Visualizations
│ └── tables/ # Results tables
├── models/ # Saved models
├── requirements.txt # Dependencies
├── .gitignore
└── README.md # This file



### Core Module: `GeometricAnalyzer`
The `src/distance_analyzer.py` module implements the technical spine with:
- **Multi-metric comparison**: Euclidean, Cosine, Cityblock, Correlation
- **Neighbor agreement analysis**: Quantifies metric disagreement
- **Visualization suite**: Publication-ready figures
- **Technical reporting**: Automated insights generation

## 🚀 Getting Started

### 1. Installation
```bash
# Clone repository
git clone https://github.com/NathanOpira/Air-Quality-Analysis.git
cd Air-Quality-Analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data processing
python -c "from src.data_loader import run_data_pipeline; run_data_pipeline()"

# Execute the distance geometry analysis (technical spine)
jupyter notebook notebooks/05_distance_geometry.ipynb

# Run all notebooks in sequence
for notebook in notebooks/0*.ipynb; do
  echo "Running $notebook..."
  jupyter nbconvert --to notebook --execute "$notebook" --inplace
done

📚 References & Resources
Academic Foundations

    Distance Geometry: Blumenthal, L. M. (1953). Theory and Applications of Distance Geometry

    Metric Learning: Weinberger, K. Q., & Saul, L. K. (2009). Distance Metric Learning for Large Margin Nearest Neighbor Classification

    Manifold Learning: Tenenbaum, J. B., et al. (2000). A Global Geometric Framework for Nonlinear Dimensionality Reduction

Technical Tools

    Scipy: Distance computation (pdist, squareform)

    Scikit-learn: Nearest neighbors, metric utilities

    Matplotlib/Seaborn: Scientific visualization

    Pandas/Numpy: Data manipulation

👤 Author

Nathan Opira
Data Science & Analytics Student | Technical Spine Practitioner
GitHub | LinkedIn
📄 License

MIT License - see LICENSE file for details.
🙏 Acknowledgements

    UCI Machine Learning Repository for the Air Quality dataset

    The distance geometry research community

    Mentors and peers who provided feedback