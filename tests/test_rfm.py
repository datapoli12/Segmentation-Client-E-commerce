"""
Unit tests for RFM feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from customer_segmentation.features.rfm import compute_rfm, scale_features


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    return pd.DataFrame({
        "CustomerID": [1, 1, 1, 2, 2, 3],
        "InvoiceNo": ["A001", "A002", "A003", "B001", "B002", "C001"],
        "InvoiceDate": pd.to_datetime([
            "2023-01-01", "2023-01-15", "2023-02-01",
            "2023-01-10", "2023-01-20",
            "2023-02-15"
        ]),
        "TotalAmount": [100.0, 150.0, 200.0, 50.0, 75.0, 300.0]
    })


def test_compute_rfm_shape(sample_transactions):
    """Test that RFM computation returns correct shape."""
    reference_date = pd.Timestamp("2023-03-01")
    rfm = compute_rfm(sample_transactions, reference_date=reference_date)

    assert rfm.shape == (3, 4)  # 3 customers, 4 columns
    assert list(rfm.columns) == ["CustomerID", "Recency", "Frequency", "Monetary"]


def test_compute_rfm_values(sample_transactions):
    """Test that RFM values are computed correctly."""
    reference_date = pd.Timestamp("2023-03-01")
    rfm = compute_rfm(sample_transactions, reference_date=reference_date)

    # Customer 1: last purchase 2023-02-01, 3 invoices, total 450
    cust1 = rfm[rfm["CustomerID"] == 1].iloc[0]
    assert cust1["Recency"] == 28  # Days from 2023-02-01 to 2023-03-01
    assert cust1["Frequency"] == 3
    assert cust1["Monetary"] == 450.0


def test_scale_features_standard():
    """Test standard scaling."""
    df = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Recency": [10, 20, 30],
        "Frequency": [1, 2, 3],
        "Monetary": [100, 200, 300]
    })

    df_scaled, scaler = scale_features(df, ["Recency", "Frequency", "Monetary"], method="standard")

    # Check that scaled values have mean ~0 and std ~1
    for col in ["Recency", "Frequency", "Monetary"]:
        assert abs(df_scaled[col].mean()) < 1e-10
        assert abs(df_scaled[col].std(ddof=0) - 1.0) < 1e-10


def test_scale_features_robust():
    """Test robust scaling."""
    df = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5],
        "Recency": [10, 20, 30, 40, 1000],  # Outlier
        "Frequency": [1, 2, 3, 4, 5],
        "Monetary": [100, 200, 300, 400, 500]
    })

    df_scaled, scaler = scale_features(df, ["Recency", "Frequency", "Monetary"], method="robust")

    # Robust scaler should be less affected by outlier
    assert df_scaled is not None
    assert "Recency" in df_scaled.columns
