from __future__ import annotations

import re

import pandas as pd


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", str(value).lower()).strip()


def _user_skills(query: str, frame: pd.DataFrame) -> set[str]:
    query_text = _normalise(query)
    known = {skill.lower(): skill for skills in frame["skill_list"] for skill in skills}
    return {original for lowered, original in known.items() if re.search(rf"\b{re.escape(lowered)}\b", query_text)}


def _structured_score(query: str, row: pd.Series) -> float:
    query_text = _normalise(query)
    signals = [row["category"], row["location"], row["employment_type"], row["experience_level"]]
    mentioned = [signal for signal in signals if signal and _normalise(signal) in query_text]
    return min(1.0, 0.5 + (0.5 * len(mentioned) / len(signals)))


def recommend_jobs(query: str, frame: pd.DataFrame, semantic_scores) -> pd.DataFrame:
    results = frame.copy()
    user_skills = _user_skills(query, frame)
    skill_scores = []
    matching = []
    missing = []
    structured = []
    reasons = []

    for _, row in results.iterrows():
        job_skills = set(row["skill_list"])
        job_lower = {skill.lower(): skill for skill in job_skills}
        matched = [job_lower[skill.lower()] for skill in user_skills if skill.lower() in job_lower]
        missing_skills = [skill for skill in job_skills if skill.lower() not in {item.lower() for item in matched}]
        skill_score = len(matched) / len(job_skills) if job_skills else 0.0
        structured_score = _structured_score(query, row)
        skill_scores.append(skill_score)
        matching.append(matched)
        missing.append(missing_skills)
        structured.append(structured_score)
        reasons.append(_reason(matched, row, query))

    results["semantic_score"] = semantic_scores
    results["skill_score"] = skill_scores
    results["structured_score"] = structured
    results["match_score"] = (0.60 * results["semantic_score"] + 0.25 * results["skill_score"] + 0.15 * results["structured_score"]) * 100
    results["matching_skills"] = matching
    results["missing_skills"] = missing
    results["match_reason"] = reasons
    return results.sort_values(["match_score", "salary_max"], ascending=[False, False]).reset_index(drop=True)


def _reason(matched: list[str], row: pd.Series, query: str) -> str:
    parts = []
    if matched:
        parts.append(f"Matches your {', '.join(matched[:3])} skill")
    if _normalise(row["experience_level"]) in _normalise(query):
        parts.append(f"matches {row['experience_level']} preference")
    if _normalise(row["location"]) in _normalise(query):
        parts.append(f"matches {row['location']}")
    return "; ".join(parts) if parts else "Strong semantic similarity across the role, skills, and description"
