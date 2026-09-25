# Pathfinder Job Finder

Pathfinder is a local, AI-assisted job finder built with Streamlit. It loads the provided job dataset dynamically, supports jobs across IT and non-IT industries, and recommends roles based on a natural-language profile.

## Features

- Dashboard KPIs for jobs, categories, locations, companies, and salary averages
- Interactive Plotly charts for category, location, employment type, experience, and salary
- Natural-language job search such as `Python developer with SQL experience`
- Combined filters for category, location, employment type, experience, and salary
- Sort by match, salary, job ID, or title
- Job cards with details, skills, descriptions, and related jobs
- AI Recommendations with semantic similarity, skill matching, missing skills, and reasons
- Transparent match score: 60% semantic similarity, 25% skill similarity, 15% structured preference match
- Cached local search index under `models/`
- Feedback page that stores user ratings and comments locally
- No API key or paid service required

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
    ├── data_loader.py
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

## Semantic Search

Each posting is converted into one searchable text document containing its title, category, skills, description, location, experience level, and employment type. The query is compared with those documents and ranked by similarity. Recommendations combine that ranking with detected skill overlap and structured preference matches.

The generated cache is stored in `models/embeddings.npz` with metadata in `models/embeddings.json`. The cache is automatically rebuilt when the CSV content changes.

Feedback submitted through the Feedback page is appended to `data/feedback.csv`.
That file is intentionally ignored by Git because it may contain optional names
and email addresses.

To rebuild it manually, delete the generated files and restart Streamlit:

```powershell
Remove-Item models\embeddings.npz, models\embeddings.json -ErrorAction SilentlyContinue
streamlit run app.py
```

## Example Queries

- `Python developer with SQL and Git`
- `Entry-level healthcare jobs in Davao`
- `Remote customer service work`
- `Marketing role with communication and sales experience`

## Error Handling

Friendly messages are shown for a missing dataset, invalid CSV schema, empty results, missing salary data, and an empty recommendation profile. Filters and charts are calculated from the loaded CSV at runtime.
