def run_data_pipeline(filepath=None):
    """
    Run the complete data loading and cleaning pipeline.

    Parameters:
    -----------
    filepath : str, optional
        Path to the raw data CSV file

    Returns:
    --------
    tuple: (cleaned_data, metadata)
    """
    print("="*60)
    print("Data Pipeline: Loading and Cleaning")
    print("="*60)

    # Load raw data
    df_raw = load_raw_data(filepath)
    print(f"✓ Raw data loaded: {df_raw.shape}")

    # Clean data
    df_clean, meta = clean_data(df_raw)
    print(f"✓ Data cleaned: {df_clean.shape}")

    # Create output directories
    import os
    os.makedirs('../data/processed', exist_ok=True)

    # Save cleaned data
    save_path = '../data/processed/air_quality_cleaned.csv'
    df_clean.to_csv(save_path)
    print(f"✓ Cleaned data saved to: {save_path}")

    # Save metadata
    import json
    meta_path = '../data/processed/cleaning_metadata.json'
    with open(meta_path, 'w') as f:
        # Convert non-serializable objects
        serializable_meta = {}
        for key, value in meta.items():
            if hasattr(value, 'isoformat'):  # Handle datetime objects
                serializable_meta[key] = value.isoformat()
            else:
                serializable_meta[key] = value
        json.dump(serializable_meta, f, indent=2)
    print(f"✓ Metadata saved to: {meta_path}")

    return df_clean, meta