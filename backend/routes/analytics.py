"""
backend/routes/analytics.py
---------------------------
Model benchmark analytics endpoint:
- GET /analytics/models: Serves comparative evaluation metrics from reports/results_test.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/analytics", tags=["analytics"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TEST_CSV = PROJECT_ROOT / "reports" / "results_test.csv"


@router.get(
    "/models",
    summary="Get model benchmark evaluation results",
)
async def get_model_benchmarks() -> list[dict[str, Any]]:
    """Return the final offline test evaluation metrics across all candidate models."""
    if not RESULTS_TEST_CSV.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test evaluation benchmark results not found.",
        )

    try:
        df = pd.read_csv(RESULTS_TEST_CSV)
        # Convert NaN values to None for clean JSON serialization
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        return records
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read benchmark results: {e}",
        ) from e
