from __future__ import annotations

import streamlit as st


MODEL_NAME = "gemini-3.5-flash"


def _api_key() -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in Streamlit secrets.")
    return api_key


def _skills(job: dict[str, object]) -> str:
    skills = job.get("skill_list", [])
    if not isinstance(skills, list):
        return str(skills)
    return ", ".join(str(skill) for skill in skills)


def _value(job: dict[str, object], key: str) -> str:
    return str(job.get(key, ""))


def _generate(prompt: str) -> str:
    from google import genai

    with genai.Client(api_key=_api_key()) as client:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text or "Gemini returned an empty response."


def generate_interview_questions(job: dict[str, object], profile: str, count: int = 5) -> str:
    prompt = f"""
Act as an interview coach. Create {count} practical questions: a balanced mix of
technical, behavioral, and role-specific questions. For each, state what a strong
answer should demonstrate. Use only requirements supported by the job description.

Job title: {_value(job, "job_title")}
Experience level: {_value(job, "experience_level")}
Skills: {_skills(job)}
Job description: {_value(job, "description")}

Candidate: {profile}
"""
    return _generate(prompt)


def evaluate_answer(job: dict[str, object], question: str, answer: str, profile: str) -> str:
    prompt = f"""
Act as an interview coach. Evaluate the answer below and return: score /10, two
specific strengths, the most important improvement, and a concise improved answer.
Be constructive; use only facts from the candidate profile or answer.

Job title: {_value(job, "job_title")}
Skills: {_skills(job)}
Candidate: {profile}
Question: {question}
Answer: {answer}
"""
    return _generate(prompt)