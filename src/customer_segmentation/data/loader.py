"""
Data loading and validation utilities.
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw Online Retail dataset.

    Args:
        filepath: Path to Excel file. Defaults to data/raw/Online Retail.xlsx

    Returns:
        pd.DataFrame: Raw dataset
    """
    if filepath is None:
        # Remonte: loader.py → data → customer_segmentation → src → project_root
        filepath = Path(__file__).parents[4] / "data" / "raw" / "Online Retail.xlsx"

    df = pd.read_excel(filepath)
    return df


def validate_dataframe(df: pd.DataFrame, expected_shape: tuple = (541909, 8)) -> dict:
    """
    Validate dataframe structure and content.

    Args:
        df: DataFrame to validate
        expected_shape: Expected (rows, cols) tuple

    Returns:
        dict: Validation report
    """
    report = {
        "shape": df.shape,
        "shape_valid": df.shape == expected_shape,
        "columns": list(df.columns),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return report
