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


def scale_features(
    df: pd.DataFrame,
    columns: list,
    method: Literal["standard", "robust"] = "robust"
) -> Tuple[pd.DataFrame, object]:
    """
    Scale RFM features.

    Args:
        df: DataFrame with RFM features
        columns: Columns to scale
        method: Scaling method ('standard' or 'robust')

    Returns:
        Tuple[pd.DataFrame, scaler]: Scaled DataFrame and fitted scaler
    """
    scaler = RobustScaler() if method == "robust" else StandardScaler()

    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])

    return df_scaled, scaler
