from __future__ import annotations

import pandas as pd


def dashboard_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "jobs": len(frame),
        "categories": frame["category"].nunique(),
        "locations": frame["location"].nunique(),
        "companies": frame["company"].nunique(),
        "average_min": frame["salary_min"].mean(),
        "average_max": frame["salary_max"].mean(),
    }


def salary_by_category(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("category", as_index=False)[["salary_min", "salary_max"]]
        .mean()
        .melt(id_vars="category", var_name="salary_type", value_name="salary")
    )
