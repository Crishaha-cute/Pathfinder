from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.analytics import dashboard_metrics, salary_by_category
from utils.chatbot import answer_question
from utils.data_loader import find_dataset, format_salary, load_jobs, non_empty_values
from utils.recommender import recommend_jobs
from utils.semantic_search import build_or_load_index


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Pathfinder | Job Finder", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #17231d; --muted: #6b766f; --paper: #f6f7f2; --lime: #d8f36a; --mint: #dfeee7; --line: #dbe2d8; --green: #2f6b4f; --orange: #b85d32; }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
    h1 { font-size: clamp(1.8rem, 3vw, 3.2rem); line-height: 1.05; }
    h2 { font-size: 1.45rem; }
    h3 { font-size: 1.08rem; }
    [data-testid='stSidebar'] { background: #edf2e9; border-right: 1px solid var(--line); }
    [data-testid='stMetric'] { background: white; border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; }
    [data-testid='stMetricValue'] { color: var(--green); font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; }
    .eyebrow { color: #758c42; font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero { background: var(--ink); color: white; border-radius: 10px; padding: 22px 26px; margin-bottom: 18px; }
    .hero h1 { color: var(--lime); margin: 4px 0 8px; }
    .hero p { color: #d8e3d6; margin: 0; max-width: 650px; }
    .topbar-brand { color: var(--green); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; padding: 8px 0; }
    .topbar-caption { color: var(--muted); font-size: .76rem; }
    [data-testid='stRadio'] label { color: var(--green); font-size: .85rem; font-weight: 600; }
    div[data-testid='stPopover'] { display: inline-flex; height: 60px; position: fixed; right: 22px; bottom: 22px; width: 60px !important; z-index: 999; }
    div[data-testid='stPopover'] > button { background: var(--green); border: 0; border-radius: 50%; color: white; font-size: 1.2rem; height: 60px !important; min-height: 60px !important; max-height: 60px; min-width: 60px !important; padding: 0 !important; width: 60px !important; }
    div[data-testid='stPopover'] > button:hover { background: #244f3a; color: white; }
    div[data-testid='stPopoverBody'] { height: 550px !important; max-height: 550px !important; min-height: 550px !important; overflow: hidden; width: 380px !important; max-width: 380px !important; min-width: 380px !important; }
    .chat-title { color: var(--green); font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; }
    .chat-welcome { background: #f1eee4; border-radius: 8px; color: var(--muted); font-size: .82rem; margin: 10px 0; padding: 10px; }
    .chat-user { background: var(--mint); border-radius: 12px 12px 3px 12px; margin: 8px 0 8px auto; max-width: 88%; padding: 8px 10px; }
    .chat-assistant { background: #f1eee4; border-radius: 12px 12px 12px 3px; margin: 8px auto 8px 0; max-width: 92%; padding: 8px 10px; }
    .chat-label { color: var(--green); display: block; font-size: .68rem; font-weight: 700; margin-bottom: 3px; text-transform: uppercase; }
    div[data-testid='stPopoverBody'] input { border: 1px solid #b9c8bd; border-radius: 7px; }
    div[data-testid='stFormSubmitButton'] button { border-radius: 50%; font-size: 1rem; height: 38px; min-height: 38px; padding: 0; width: 38px; }
    [data-testid='stVerticalBlockBorderWrapper'] { background: white; border-color: var(--line); border-radius: 8px; padding: 10px 14px; }
    .job-card h3, [data-testid='stVerticalBlockBorderWrapper'] h3 { color: var(--green); margin: 3px 0; }
    .job-meta { color: #527b70; font-size: .8rem; margin: 2px 0; }
    .salary { color: var(--orange); font-weight: 700; margin: 8px 0; }
    .badge { background: var(--mint); border-radius: 999px; color: #315642; display: inline-block; font-size: .7rem; margin: 2px 2px 0 0; padding: 3px 7px; }
    .match { color: var(--green); font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; }
    .small-note { color: var(--muted); font-size: .76rem; }
    div.stButton > button { border-radius: 6px; border: 1px solid var(--ink); font-size: .82rem; font-weight: 600; padding: 5px 12px; }
    div.stButton > button[kind='primary'] { background: var(--lime); color: var(--ink); border-color: var(--lime); }
    @media (max-width: 700px) { .hero { padding: 22px; } .job-card { min-height: 0; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_jobs(dataset_path: str) -> pd.DataFrame:
    return load_jobs(Path(dataset_path))


@st.cache_resource
def get_index(frame: pd.DataFrame):
    return build_or_load_index(frame, MODELS_DIR / "embeddings.npz")


def money(value: object) -> str:
    return "Not listed" if pd.isna(value) else f"{float(value):,.0f}"


dataset_path = find_dataset(PROJECT_ROOT)
if dataset_path is None:
    st.error("No dataset found. Add data/job_postings_128.csv to continue.")
    st.stop()

try:
    jobs = get_jobs(str(dataset_path))
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load the job dataset: {error}")
    st.stop()

if jobs.empty:
    st.error("The dataset has no usable job postings.")
    st.stop()

index = get_index(jobs)


for state_key, default_value in {
    "search_query": "",
    "filter_categories": [],
    "filter_locations": [],
    "filter_types": [],
    "filter_experience": [],
    "salary_range": "Any salary",
    "sort_option": "Best Match",
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


def clear_filters(salary_maximum: int) -> None:
    for key in ["filter_categories", "filter_locations", "filter_types", "filter_experience"]:
        st.session_state[key] = []
    st.session_state["salary_range"] = "Any salary"


def clear_search_state(salary_maximum: int) -> None:
    st.session_state["search_query"] = ""
    st.session_state["sort_option"] = "Best Match"
    st.session_state["search_query_widget"] = ""
    st.session_state["sort_option_widget"] = "Best Match"
    clear_filters(salary_maximum)


def sync_search_state() -> None:
    st.session_state["search_query"] = st.session_state["search_query_widget"]
    st.session_state["sort_option"] = st.session_state["sort_option_widget"]


header_brand, header_navigation = st.columns([1, 3])
with header_brand:
    st.markdown("<div class='topbar-brand'>◈ Pathfinder</div><div class='topbar-caption'>Find work that fits.</div>", unsafe_allow_html=True)
with header_navigation:
    page = st.radio(
        "Navigate",
        ["Dashboard", "Find Jobs", "AI Recommendations", "Analytics"],
        horizontal=True,
        key="page",
        label_visibility="collapsed",
    )


with st.sidebar:
    st.markdown("### Refine results")
    categories = list(non_empty_values(jobs, "category"))
    locations = list(non_empty_values(jobs, "location"))
    employment_types = list(non_empty_values(jobs, "employment_type"))
    experience_levels = list(non_empty_values(jobs, "experience_level"))
    selected_categories = st.multiselect("Category", categories, key="filter_categories")
    selected_locations = st.multiselect("Location", locations, key="filter_locations")
    selected_types = st.multiselect("Employment type", employment_types, key="filter_types")
    selected_experience = st.multiselect("Experience level", experience_levels, key="filter_experience")
    salary_maximum = int(jobs["salary_max"].max()) if jobs["salary_max"].notna().any() else 0
    salary_options = ["Any salary", "Below 20,000", "20,000 - 39,999", "40,000 - 59,999", "60,000+"]
    selected_salary = st.selectbox("Salary range", salary_options, key="salary_range")
    salary_bounds = {
        "Any salary": (0, salary_maximum),
        "Below 20,000": (0, 19_999),
        "20,000 - 39,999": (20_000, 39_999),
        "40,000 - 59,999": (40_000, 59_999),
        "60,000+": (60_000, salary_maximum),
    }
    minimum_salary, maximum_salary = salary_bounds[selected_salary]
    st.session_state["min_salary"] = minimum_salary
    st.session_state["max_salary"] = maximum_salary
    st.button("Clear filters", use_container_width=True, on_click=clear_filters, args=(salary_maximum,))


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    if selected_categories:
        result = result[result["category"].isin(selected_categories)]
    if selected_locations:
        result = result[result["location"].isin(selected_locations)]
    if selected_types:
        result = result[result["employment_type"].isin(selected_types)]
    if selected_experience:
        result = result[result["experience_level"].isin(selected_experience)]
    if salary_maximum:
        result = result[result["salary_min"].fillna(0).between(minimum_salary, maximum_salary)]
    return result.copy()


def semantic_scores(frame: pd.DataFrame, query: str) -> pd.Series:
    scores_by_job_id = pd.Series(index.search(query), index=jobs["job_id"].astype(str))
    return frame["job_id"].astype(str).map(scores_by_job_id).fillna(0.0)


filtered_jobs = apply_filters(jobs)


def render_job_card(row: pd.Series, score: float | None = None, key_prefix: str = "job") -> None:
    with st.container(border=True):
        top_left, top_right = st.columns([4, 1])
        with top_left:
            st.markdown(f"<div class='eyebrow'>{row['category']}</div><h3>{row['job_title']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='job-meta'>{row['company']} · {row['location']}</div>", unsafe_allow_html=True)
        with top_right:
            if score is not None:
                st.markdown(f"<div class='match'>{score:.0f}%</div><div class='small-note'>profile match</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='salary'>{format_salary(row['salary_min'], row['salary_max'])}</div>", unsafe_allow_html=True)
        badges = "".join(f"<span class='badge'>{skill}</span>" for skill in row["skill_list"][:5])
        st.markdown(badges or "<span class='small-note'>Skills not listed</span>", unsafe_allow_html=True)
        st.caption(row["description"][:170] + ("..." if len(row["description"]) > 170 else ""))
        if st.button("View details", key=f"{key_prefix}_{row['job_id']}", type="primary"):
            show_job_details(row["job_id"])


@st.dialog("Job details", width="large")
def show_job_details(job_id: str) -> None:
    selected = jobs[jobs["job_id"] == job_id]
    if selected.empty:
        st.error("This job is no longer available.")
        return
    row = selected.iloc[0]
    st.markdown(f"<div class='eyebrow'>{row['category']}</div>", unsafe_allow_html=True)
    st.header(row["job_title"])
    st.markdown(f"**{row['company']}** · {row['location']} · {row['employment_type']} · {row['experience_level']}")
    st.markdown(f"### Salary: {format_salary(row['salary_min'], row['salary_max'])}")
    if row["application_url"] or row["company_website"]:
        apply_column, website_column = st.columns(2)
        with apply_column:
            if row["application_url"]:
                st.link_button("Apply Now", row["application_url"], use_container_width=True)
        with website_column:
            if row["company_website"]:
                st.link_button("Company Website", row["company_website"], use_container_width=True)
    contact_items = []
    if row["company_email"]:
        contact_items.append(f"[Email employer](mailto:{row['company_email']})")
    if row["company_phone"]:
        contact_items.append(f"[Call {row['employer_name']}](tel:{row['company_phone']})")
    if contact_items:
        st.markdown(f"**Contact {row['employer_name']}**  \n{' · '.join(contact_items)}")
    st.markdown("### Required skills")
    st.write(", ".join(row["skill_list"]) or "Skills not listed")
    st.markdown("### About the role")
    st.write(row["description"])
    related = jobs[jobs["job_id"] != row["job_id"]].copy()
    if not related.empty:
        related_scores = index.search(row["search_text"])
        related["similarity"] = related_scores[related.index]
        st.markdown("### Related jobs")
        for _, related_row in related.nlargest(3, "similarity").iterrows():
            st.markdown(f"- **{related_row['job_title']}** at {related_row['company']} · {related_row['location']}")


def render_dashboard() -> None:
    st.markdown("<div class='hero'><div class='eyebrow'>JOB MARKET / 2026</div><h1>Find work that fits.</h1><p>Explore opportunities across industries, then use your own words to find roles that make sense for you.</p></div>", unsafe_allow_html=True)
    metrics = dashboard_metrics(filtered_jobs)
    labels = [("Jobs", metrics["jobs"]), ("Categories", metrics["categories"]), ("Locations", metrics["locations"]), ("Companies", metrics["companies"]), ("Avg. min salary", money(metrics["average_min"])), ("Avg. max salary", money(metrics["average_max"]))]
    for row_start in range(0, len(labels), 3):
        metric_columns = st.columns(3)
        for column, (label, value) in zip(metric_columns, labels[row_start:row_start + 3]):
            column.metric(label, value)
    if filtered_jobs.empty:
        st.info("No jobs match the current filters. Clear or widen the filters to see the market charts.")
        return
    st.markdown("### Market snapshot")
    left, right = st.columns(2)
    with left:
        category_counts = filtered_jobs["category"].value_counts().reset_index(name="jobs")
        category_counts.columns = ["category", "jobs"]
        st.plotly_chart(px.bar(category_counts, x="jobs", y="category", orientation="h", color="jobs", color_continuous_scale="YlGn", title="Jobs by category"), use_container_width=True)
    with right:
        location_counts = filtered_jobs["location"].value_counts().head(12).reset_index(name="jobs")
        location_counts.columns = ["location", "jobs"]
        st.plotly_chart(px.bar(location_counts, x="jobs", y="location", orientation="h", title="Top locations"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        type_counts = filtered_jobs["employment_type"].value_counts().reset_index(name="jobs")
        type_counts.columns = ["employment_type", "jobs"]
        st.plotly_chart(px.pie(type_counts, names="employment_type", values="jobs", hole=.48, title="Employment type"), use_container_width=True)
    with right:
        experience_counts = filtered_jobs["experience_level"].value_counts().reset_index(name="jobs")
        experience_counts.columns = ["experience_level", "jobs"]
        st.plotly_chart(px.bar(experience_counts, x="experience_level", y="jobs", title="Experience level"), use_container_width=True)


def render_find_jobs() -> None:
    st.markdown("<div class='eyebrow'>DISCOVER</div>", unsafe_allow_html=True)
    st.title("Find your next role")
    st.session_state["search_query_widget"] = st.session_state["search_query"]
    st.session_state["sort_option_widget"] = st.session_state["sort_option"]
    search_column, sort_column, clear_column = st.columns([2.4, 1.35, .75], vertical_alignment="bottom")
    with search_column:
        query = st.text_input("Search jobs", placeholder="Try: Python developer in Davao", key="search_query_widget", on_change=sync_search_state)
    with sort_column:
        sort_by = st.selectbox("Sort results", ["Best Match", "Salary: Highest to Lowest", "Salary: Lowest to Highest", "Newest / Job ID", "Job Title A-Z"], key="sort_option_widget", on_change=sync_search_state)
    with clear_column:
        st.button("Clear Search", help="Clear search, filters, and sorting", on_click=clear_search_state, args=(salary_maximum,), use_container_width=True)
    results = filtered_jobs.copy()
    if query.strip():
        results["search_score"] = semantic_scores(results, query).to_numpy()
        results = results.sort_values("search_score", ascending=False)
    if sort_by == "Salary: Highest to Lowest":
        results = results.sort_values("salary_max", ascending=False, na_position="last")
    elif sort_by == "Salary: Lowest to Highest":
        results = results.sort_values("salary_min", ascending=True, na_position="last")
    elif sort_by == "Newest / Job ID":
        results = results.sort_values("job_id", ascending=False)
    elif sort_by == "Job Title A-Z":
        results = results.sort_values("job_title")
    st.caption(f"{len(results)} roles found · semantic search is based on the role, skills, description, and preferences")
    if results.empty:
        st.info("No jobs match those filters. Try widening your salary range or clearing a filter.")
        return
    for start in range(0, len(results), 2):
        columns = st.columns(2)
        for column, (_, row) in zip(columns, results.iloc[start:start + 2].iterrows()):
            with column:
                render_job_card(row, key_prefix="search")


def render_recommendations() -> None:
    st.markdown("<div class='eyebrow'>PERSONAL FIT</div>", unsafe_allow_html=True)
    st.title("Find Jobs That Match Me")
    st.write("Describe your skills, background, and ideal role. Pathfinder compares your profile with the dataset using semantic similarity, skills, and structured preferences.")
    profile = st.text_area("Your profile", placeholder="I am a computer science student who knows Python, SQL, Git and web development. I am looking for an entry-level remote job.", height=130)
    if st.button("Find matching jobs", type="primary", disabled=not profile.strip()):
        st.session_state["recommendation_query"] = profile
    profile = st.session_state.get("recommendation_query", profile)
    if not profile.strip():
        st.info("Add a profile description to see recommendations.")
        return
    candidates = apply_filters(jobs)
    recommendations = recommend_jobs(profile, candidates, semantic_scores(candidates, profile).to_numpy())
    st.caption("Match scores measure similarity to your profile. They are not hiring probabilities.")
    if recommendations.empty:
        st.info("No matching jobs are available with the current filters.")
        return
    for _, row in recommendations.head(8).iterrows():
        render_job_card(row, row["match_score"], key_prefix="recommendation")
        match_col, missing_col, reason_col = st.columns([1, 1, 2])
        with match_col:
            st.progress(min(1.0, float(row["skill_score"])), text=f"Skill match: {row['skill_score'] * 100:.0f}%")
        with missing_col:
            st.markdown(f"**Missing skills**  \n{', '.join(row['missing_skills'][:5]) or 'None listed'}")
        with reason_col:
            st.markdown(f"**Why this matches**  \n{row['match_reason']}")


def render_analytics() -> None:
    st.markdown("<div class='eyebrow'>ANALYTICS</div>", unsafe_allow_html=True)
    st.title("Salary and market detail")
    salary_frame = salary_by_category(filtered_jobs).dropna(subset=["salary"])
    if salary_frame.empty:
        st.info("Salary data is not available for the selected jobs.")
    else:
        st.plotly_chart(px.bar(salary_frame, x="category", y="salary", color="salary_type", barmode="group", title="Average salary by category"), use_container_width=True)
    st.dataframe(filtered_jobs[["job_title", "company", "category", "location", "employment_type", "experience_level", "salary_min", "salary_max"]], use_container_width=True, hide_index=True)


def render_chatbot() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    with st.popover("💬", help="Ask Pathfinder about the job dataset"):
        st.markdown("<div class='chat-title'>Pathfinder Assistant</div>", unsafe_allow_html=True)
        st.caption("Ask about roles, skills, salaries, or locations.")
        with st.container(height=355, border=False):
            if not st.session_state["chat_messages"]:
                st.markdown("<div class='chat-welcome'>Hi! I can help you explore the jobs in this dataset. Try asking, <b>Which jobs need Python?</b></div>", unsafe_allow_html=True)
            for sender, message in st.session_state["chat_messages"][-8:]:
                style = "chat-user" if sender == "user" else "chat-assistant"
                label = "You" if sender == "user" else "Pathfinder"
                st.markdown(f"<div class='{style}'><span class='chat-label'>{label}</span>{escape(message)}</div>", unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            input_column, submit_column = st.columns([6, 1], vertical_alignment="bottom")
            with input_column:
                question = st.text_input("Message", placeholder="Ask about the jobs...", label_visibility="collapsed")
            with submit_column:
                submitted = st.form_submit_button("➤", type="primary", help="Send message")
        if submitted and question.strip():
            st.session_state["chat_messages"].append(("user", question.strip()))
            scores = semantic_scores(jobs, question).to_numpy()
            response = answer_question(question, jobs, scores)
            st.session_state["chat_messages"].append(("assistant", response))
            st.rerun()


if page == "Dashboard":
    render_dashboard()
elif page == "Find Jobs":
    render_find_jobs()
elif page == "AI Recommendations":
    render_recommendations()
else:
    render_analytics()

render_chatbot()
