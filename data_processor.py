"""
Data processing module for CSV cleaning, statistics, and anomaly detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Handles data cleaning, statistics generation, and anomaly detection."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        self.original_df = df.copy()
        self.cleaned_df = None
        self.cleaning_report = {}
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by:
        - Removing duplicate rows
        - Handling missing values
        - Removing leading/trailing whitespace from string columns
        - Converting numeric columns to appropriate types
        """
        df = self.original_df.copy()
        cleaning_steps = []
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows")
        
        # Remove leading/trailing whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace empty strings with NaN
            df[col] = df[col].replace(['', 'nan', 'None'], np.nan)
        
        # Try to convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    # If at least some values are numeric, convert the column
                    non_null_before = df[col].notna().sum()
                    df[col] = numeric_series
                    non_null_after = df[col].notna().sum()
                    if non_null_before != non_null_after:
                        cleaning_steps.append(
                            f"Converted column '{col}' to numeric (lost {non_null_before - non_null_after} non-numeric values)"
                        )
        
        # Handle missing values - fill numeric columns with median, categorical with mode
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_info[col] = int(missing_count)
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col].fillna(median_val, inplace=True)
                        cleaning_steps.append(
                            f"Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})"
                        )
                else:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                        cleaning_steps.append(
                            f"Filled {missing_count} missing values in '{col}' with mode"
                        )
        
        self.cleaned_df = df
        self.cleaning_report = {
            "initial_rows": int(initial_rows),
            "final_rows": int(len(df)),
            "duplicates_removed": int(duplicates_removed),
            "missing_values_by_column": missing_info,
            "cleaning_steps": cleaning_steps
        }
        
        return df
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics for the cleaned data."""
        if self.cleaned_df is None:
            self.clean_data()
        
        df = self.cleaned_df
        stats = {
            "dataset_overview": {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "column_names": df.columns.tolist(),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
            },
            "column_statistics": {}
        }
        
        # Statistics for each column
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "null_percentage": float((df[col].isna().sum() / len(df)) * 100)
            }
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "mean": float(df[col].mean()) if df[col].notna().any() else None,
                    "median": float(df[col].median()) if df[col].notna().any() else None,
                    "std": float(df[col].std()) if df[col].notna().any() else None,
                    "min": float(df[col].min()) if df[col].notna().any() else None,
                    "max": float(df[col].max()) if df[col].notna().any() else None,
                    "q25": float(df[col].quantile(0.25)) if df[col].notna().any() else None,
                    "q75": float(df[col].quantile(0.75)) if df[col].notna().any() else None,
                    "skewness": float(df[col].skew()) if df[col].notna().any() else None,
                    "kurtosis": float(df[col].kurtosis()) if df[col].notna().any() else None
                })
            else:
                # Categorical statistics
                value_counts = df[col].value_counts()
                col_stats.update({
                    "unique_count": int(df[col].nunique()),
                    "most_frequent_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                    "top_5_values": value_counts.head(5).to_dict() if len(value_counts) > 0 else {}
                })
            
            stats["column_statistics"][col] = col_stats
        
        return stats
    
    def detect_anomalies(self, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.1)
        
        Returns:
            Dictionary with anomaly detection results
        """
        if self.cleaned_df is None:
            self.clean_data()
        
        df = self.cleaned_df
        
        # Select only numeric columns for anomaly detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return {
                "anomalies_detected": 0,
                "anomaly_percentage": 0.0,
                "anomaly_indices": [],
                "message": "No numeric columns found for anomaly detection"
            }
        
        # Prepare data for anomaly detection
        X = df[numeric_cols].values
        
        # Handle any remaining NaN values
        if np.isnan(X).any():
            # Fill with column means
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.isnan(X)
            X[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        
        # Anomalies are labeled as -1
        anomaly_mask = anomaly_labels == -1
        anomaly_indices = df.index[anomaly_mask].tolist()
        num_anomalies = int(anomaly_mask.sum())
        
        # Get anomaly scores (lower scores indicate more anomalous)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        # Create detailed anomaly report
        anomaly_details = []
        for idx in anomaly_indices[:100]:  # Limit to first 100 for response size
            row_data = df.loc[idx].to_dict()
            anomaly_details.append({
                "row_index": int(idx),
                "anomaly_score": float(anomaly_scores[df.index.get_loc(idx)]),
                "row_data": {k: (float(v) if pd.api.types.is_numeric_dtype(type(v)) else str(v)) 
                            for k, v in row_data.items()}
            })
        
        return {
            "anomalies_detected": num_anomalies,
            "anomaly_percentage": float((num_anomalies / len(df)) * 100),
            "anomaly_indices": [int(idx) for idx in anomaly_indices],
            "contamination_parameter": contamination,
            "numeric_columns_used": numeric_cols,
            "anomaly_details": anomaly_details[:50],  # Limit details to 50 for response size
            "total_anomaly_details": len(anomaly_details)
        }

