"""
RFM (Recency, Frequency, Monetary) feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Literal


def compute_rfm(
    df: pd.DataFrame,
    customer_col: str = "CustomerID",
    date_col: str = "InvoiceDate",
    invoice_col: str = "InvoiceNo",
    amount_col: str = "TotalAmount",
    reference_date: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Compute RFM features aggregated by customer.

    Args:
        df: Cleaned transaction dataframe
        customer_col: Customer identifier column
        date_col: Transaction date column
        invoice_col: Invoice identifier column
        amount_col: Transaction amount column
        reference_date: Reference date for recency calculation

    Returns:
        pd.DataFrame: RFM features indexed by customer
    """
    if reference_date is None:
        reference_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        invoice_col: "nunique",  # Frequency
        amount_col: "sum"  # Monetary
    }).reset_index()

    rfm.columns = [customer_col, "Recency", "Frequency", "Monetary"]

    return rfm


def winsorize(
    df: pd.DataFrame,
    columns: list,
    percentile: float = 95
) -> pd.DataFrame:
    """
    Cap outliers at specified percentile.

    Args:
        df: DataFrame with features
        columns: Columns to winsorize
        percentile: Upper percentile for capping (e.g., 95 for P95)

    Returns:
        pd.DataFrame: DataFrame with capped values
    """
    df_winsor = df.copy()
    for col in columns:
        cap_value = np.percentile(df[col], percentile)
        df_winsor[col] = np.clip(df[col], None, cap_value)
    return df_winsor


def scale_features(
    df: pd.DataFrame,
    columns: list,
    method: Literal["standard", "robust"] = "robust",
    winsorize_percentile: float = None
) -> Tuple[pd.DataFrame, object]:
    """
    Scale RFM features with optional winsorization.

    Args:
        df: DataFrame with RFM features
        columns: Columns to scale
        method: Scaling method ('standard' or 'robust')
        winsorize_percentile: If set, cap outliers at this percentile before scaling

    Returns:
        Tuple[pd.DataFrame, scaler]: Scaled DataFrame and fitted scaler
    """
    df_scaled = df.copy()

    # Winsorization si spécifié
    if winsorize_percentile is not None:
        df_scaled = winsorize(df_scaled, columns, winsorize_percentile)

    scaler = RobustScaler() if method == "robust" else StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

    return df_scaled, scaler
