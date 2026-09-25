from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "job_id",
    "job_title",
    "company",
    "category",
    "location",
    "employment_type",
    "experience_level",
    "salary_min",
    "salary_max",
    "skills",
    "description",
}

DATA_CANDIDATES = ("job_postings_128.csv", "job_descriptions.csv")
CONTACT_COLUMNS = ("employer_name", "company_email", "company_phone", "company_website", "application_url")


def find_dataset(project_root: Path) -> Path | None:
    for filename in DATA_CANDIDATES:
        path = project_root / "data" / filename
        if path.exists():
            return path
    return None


def _clean_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_jobs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"The CSV is missing required columns: {missing_text}")

    frame = frame.copy()
    frame["job_id"] = frame["job_id"].astype(str).str.strip()
    for column in ["job_title", "company", "category", "location", "employment_type", "experience_level", "skills", "description"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    for column in CONTACT_COLUMNS:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["employer_name"] = frame["employer_name"].where(frame["employer_name"].ne(""), frame["company"])
    frame["salary_min"] = pd.to_numeric(frame["salary_min"], errors="coerce")
    frame["salary_max"] = pd.to_numeric(frame["salary_max"], errors="coerce")
    frame["skill_list"] = frame["skills"].apply(_clean_list)
    frame["search_text"] = frame.apply(_search_text, axis=1)
    frame = frame.drop_duplicates(subset=["job_id"])
    frame = frame[frame["job_title"].ne("") & frame["description"].ne("")]
    return frame.reset_index(drop=True)


def _search_text(row: pd.Series) -> str:
    return " ".join(
        [
            str(row["job_title"]),
            str(row["category"]),
            str(row["skills"]),
            str(row["description"]),
            str(row["location"]),
            str(row["experience_level"]),
            str(row["employment_type"]),
        ]
    )


def format_salary(minimum: object, maximum: object) -> str:
    if pd.isna(minimum) and pd.isna(maximum):
        return "Salary not listed"
    if pd.isna(minimum):
        return f"Up to {float(maximum):,.0f}"
    if pd.isna(maximum):
        return f"From {float(minimum):,.0f}"
    return f"{float(minimum):,.0f} - {float(maximum):,.0f}"


def non_empty_values(frame: pd.DataFrame, column: str) -> Iterable[str]:
    return sorted(value for value in frame[column].dropna().unique() if str(value).strip())
