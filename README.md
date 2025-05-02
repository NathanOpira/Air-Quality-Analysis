# Air Quality Analysis and Prediction

## Overview
This project aims to analyze and predict air quality using data from the UCI Machine Learning Repository. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling.

## Project Structure
- **data/**: Contains raw and processed datasets.
  - `raw/`: Original datasets (e.g., `AirQualityUCI.csv`).
  - `processed/`: Cleaned and transformed datasets.
- **models/**: Saved models and related files.
- **notebooks/**: Jupyter notebooks for EDA, preprocessing, modeling, and forecasting.
- **outputs/**: Generated figures and reports.
- **src/**: Source code for data loading, feature engineering, and model training.
  - `data_loader.py`: Functions for loading and preprocessing data.
  - `features.py`: Feature engineering utilities.
  - `train_model.py`: Scripts for training predictive models.
  - `utils.py`: Helper functions.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Air-Quality-Analysis-and-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks or scripts as needed.

## Usage
- Use the `notebooks/` directory to explore the data and build models.
- Use the `src/` directory for running scripts programmatically.

## Data Source
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

## License
This project is licensed under the MIT License.