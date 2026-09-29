# Pathfinder Job Finder

Pathfinder AI-assisted job finder built with Streamlit. It loads the provided job dataset dynamically, supports jobs across IT and non-IT industries, and recommends roles based on a natural-language profile.

## Target Users

Pathfinder is designed for:

- Students and recent graduates exploring entry-level opportunities
- Job seekers searching across industries, locations, salaries, and experience levels
- Career changers identifying roles that match their transferable skills
- Applicants preparing for interviews with the Gemini Interview Coach
- Career counselors, educators, and training programs guiding job preparation
- Recruiters and workforce teams reviewing job market patterns and salary data

## Features

- Dashboard KPIs for jobs, categories, locations, companies, and salary averages
- Interactive Plotly charts for category, location, employment type, experience, and salary
- Natural-language job search such as `Python developer with SQL experience`
- Combined filters for category, location, employment type, experience, and salary
- Sort by match, salary, job ID, or title
- Job cards with details, skills, descriptions, and related jobs
- AI Recommendations with semantic similarity, skill matching, missing skills, and reasons
- Gemini Interview Coach with role-specific mock questions and answer feedback
- Pathfinder Assistant chatbot for questions about jobs, skills, salaries, and locations
- Transparent match score: 60% semantic similarity, 25% skill similarity, 15% structured preference match
- Cached local search index under `models/`
- Feedback page that stores user ratings and comments locally


A match score is a similarity score between a profile and a job posting. It is not a hiring probability.

## Project Structure

```text
JobInsightsAI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── job_postings_128.csv       # preferred filename
│   └── job_descriptions.csv       # supported fallback filename
├── models/                        # generated embedding/index cache
└── utils/
    ├── analytics.py
    ├── chatbot.py
    ├── data_loader.py
    ├── feedback.py
    ├── interview_coach.py
    ├── recommender.py
    └── semantic_search.py
```

## Dataset

The preferred file is `data/job_postings_128.csv`. The current workspace's `data/job_descriptions.csv` is also supported. The required columns are:

```text
job_id, job_title, company, category, location, employment_type,
experience_level, salary_min, salary_max, skills, description
```

Contact columns supported by the job details popup are `employer_name`,
`company_email`, `company_phone`, `company_website`, and `application_url`.
They are optional; when present, the app shows Apply Now, company website,
email, and phone links.

The application validates these columns, converts invalid salary values to missing values, removes duplicate IDs, and ignores postings without a title or description. Job postings are never hard-coded in the application.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

The first installation includes `sentence-transformers` and may download `all-MiniLM-L6-v2` when that model is first used. After the model is downloaded, the app can run offline. If the embedding package or model is unavailable, Pathfinder automatically uses a cached NumPy TF-IDF semantic index so the application remains usable.

## Example Queries

- `Python developer with SQL and Git`
- `Entry-level healthcare jobs in Davao`
- `Remote customer service work`
- `Marketing role with communication and sales experience`

## Error Handling

Friendly messages are shown for a missing dataset, invalid CSV schema, empty results, missing salary data, and an empty recommendation profile. Filters and charts are calculated from the loaded CSV at runtime.
