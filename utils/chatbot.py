from __future__ import annotations

import re

import pandas as pd

from utils.data_loader import format_salary


def answer_question(question: str, jobs: pd.DataFrame, scores) -> str:
    query = question.strip().lower()
    if not query:
        return "Ask me about the jobs, salaries, skills, locations, or roles in this dataset."

    known_skills = sorted({skill for skills in jobs["skill_list"] for skill in skills}, key=len, reverse=True)
    requested_skills = [skill for skill in known_skills if re.search(rf"\b{re.escape(skill.lower())}\b", query)]
    if requested_skills:
        matching_jobs = jobs[jobs["skill_list"].apply(lambda skills: any(skill.lower() in {item.lower() for item in skills} for skill in requested_skills))]
        if matching_jobs.empty:
            return f"I could not find postings that list {', '.join(requested_skills)} as a required skill."
        titles = "; ".join(f"{row['job_title']} at {row['company']}" for _, row in matching_jobs.head(5).iterrows())
        return f"I found {len(matching_jobs)} jobs requiring {', '.join(requested_skills)}. Examples: {titles}."

    if re.search(r"how many|number of|total", query) and "categor" not in query and "compan" not in query and "location" not in query:
        return f"There are {len(jobs)} job postings in the dataset."

    if "categor" in query or "industr" in query:
        counts = jobs["category"].value_counts().head(5)
        summary = ", ".join(f"{category} ({count})" for category, count in counts.items())
        return f"There are {jobs['category'].nunique()} categories. The largest are {summary}."

    if "compan" in query:
        counts = jobs["company"].value_counts().head(5)
        summary = ", ".join(f"{company} ({count})" for company, count in counts.items())
        return f"The dataset includes {jobs['company'].nunique()} companies. The most represented are {summary}."

    if "where" in query or "location" in query or "city" in query:
        counts = jobs["location"].value_counts().head(5)
        summary = ", ".join(f"{location} ({count})" for location, count in counts.items())
        return f"The most common job locations are {summary}."

    if "salary" in query or "pay" in query or "earning" in query:
        minimum = jobs["salary_min"].mean()
        maximum = jobs["salary_max"].mean()
        return f"Across listed jobs, the average salary range is {format_salary(minimum, maximum)}. Individual salaries vary by role and location."

    if "skill" in query or "know" in query or "learn" in query:
        skills = [skill for row in jobs["skill_list"] for skill in row]
        counts = pd.Series(skills).value_counts().head(8)
        summary = ", ".join(f"{skill} ({count})" for skill, count in counts.items())
        return f"The most requested skills are {summary}."

    if "remote" in query:
        remote_jobs = jobs[jobs["location"].str.contains("remote", case=False, na=False)]
        return f"I found {len(remote_jobs)} remote postings. Try asking me about a specific skill or role to narrow them down."

    ranked = jobs.copy()
    ranked["similarity"] = scores
    ranked = ranked.nlargest(3, "similarity")
    if ranked.empty:
        return "I could not find a matching job in the current dataset."
    recommendations = "; ".join(
        f"{row['job_title']} at {row['company']} ({row['location']})" for _, row in ranked.iterrows()
    )
    return f"Based on your question, these roles look most relevant: {recommendations}. Ask me for details about any of them."
