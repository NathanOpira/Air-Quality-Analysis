"""Feature engineering utilities for Air Quality Analysis.

Provides `create_analytical_features` which creates a set of analytical
features from raw air-quality DataFrames for modeling and geometric analysis.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def create_analytical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
	"""
	Create features for geometric analysis and modeling.

	Returns a tuple with (features_dataframe, feature_names).

	The function will:
	- keep a copy of the input
	- identify pollution and sensor cols (columns containing '(GT)' or
	  commonly used sensor names 'T', 'RH', 'AH')
	- add normalized pollution columns (suffix '_norm')
	- add cyclic temporal encodings for hour and day-of-week when a
	  datetime index is present (or if a 'date' column is present it will
	  attempt to convert it)
	- add pollutant ratios and fractions (CO/NOx, NO2/NOx)
	- add first-differences for short-term rate-of-change
	- add simple rolling statistics and lag features
	- add a few interaction terms (temperature x humidity)

	Parameters
	----------
	df : pd.DataFrame
		Input time-indexed or indexed dataset containing pollutant and
		environmental sensor columns.

	Returns
	-------
	Tuple[pd.DataFrame, List[str]]
		DataFrame containing engineered features (aligned with input index)
		and the ordered list of feature column names.
	"""

	# Make a working copy
	df_feat = df.copy()

	# If user supplied a 'date' column but index is not datetime, try to use it
	if not pd.api.types.is_datetime64_any_dtype(df_feat.index) and "date" in df_feat.columns:
		try:
			df_feat.index = pd.to_datetime(df_feat["date"]).tz_localize(None)
		except Exception:
			pass

	# Identify pollution/sensor features
	pollution_features = [col for col in df_feat.columns if "(GT)" in col or col in ["T", "RH", "AH"]]

	# 1. Original pollution features (normalized -> new columns with _norm)
	for col in pollution_features:
		try:
			series = df_feat[col].astype(float)
			mean = series.mean(skipna=True)
			std = series.std(skipna=True)
			if pd.isna(std) or std == 0:
				df_feat[f"{col}_norm"] = series.fillna(0)
			else:
				df_feat[f"{col}_norm"] = (series - mean) / std
		except Exception:
			df_feat[f"{col}_norm"] = df_feat[col]

	# 2. Temporal features (cyclic encoding)
	if pd.api.types.is_datetime64_any_dtype(df_feat.index):
		idx = df_feat.index
		df_feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
		df_feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
		df_feat["day_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
		df_feat["day_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
		df_feat["is_weekend"] = (idx.dayofweek >= 5).astype(int)

	# 3. Pollutant ratios (combustion signatures)
	if "CO(GT)" in df_feat.columns and "NOx(GT)" in df_feat.columns:
		df_feat["CO_to_NOx"] = df_feat["CO(GT)"] / (df_feat["NOx(GT)"] + 1e-10)

	if "NO2(GT)" in df_feat.columns and "NOx(GT)" in df_feat.columns:
		df_feat["NO2_fraction"] = df_feat["NO2(GT)"] / (df_feat["NOx(GT)"] + 1e-10)

	# 4. Rate of change (1st derivative) for the first up-to-5 pollutants
	for col in pollution_features[:5]:
		if col in df_feat.columns:
			df_feat[f"{col}_diff"] = df_feat[col].diff().fillna(0)

	# 5. Rolling statistics and lag features
	rolling_windows = [3, 24]
	for w in rolling_windows:
		for col in pollution_features:
			if col in df_feat.columns:
				df_feat[f"{col}_roll_mean_{w}"] = df_feat[col].rolling(window=w, min_periods=1).mean()
				df_feat[f"{col}_roll_std_{w}"] = df_feat[col].rolling(window=w, min_periods=1).std().fillna(0)

	# Add simple lags (1-hour and 24-hour) if appropriate
	lags = [1, 24]
	for lag in lags:
		for col in pollution_features:
			if col in df_feat.columns:
				df_feat[f"{col}_lag_{lag}"] = df_feat[col].shift(lag)

	# 6. Interaction terms (temperature x humidity)
	if "T" in df_feat.columns and "RH" in df_feat.columns:
		df_feat["T_x_RH"] = df_feat["T"] * df_feat["RH"]

	# 7. Fallback missing-value handling (do not drop index)
	df_feat = df_feat.fillna(method="ffill").fillna(0)

	# 8. Build feature column list: include normalized pollution features first,
	#    then the engineered features sorted deterministically
	norm_cols = [f"{c}_norm" for c in pollution_features if f"{c}_norm" in df_feat.columns]
	engineered_cols = [c for c in df_feat.columns if c not in pollution_features and c not in norm_cols]

	feature_cols = pollution_features + norm_cols + engineered_cols

	return df_feat.loc[:, feature_cols], feature_cols

