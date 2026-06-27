"""
Legend Acers — Intelligent Candidate Discovery
===============================================
Streamlit entry point for HuggingFace Spaces deployment.
Entry point must be app.py at repo root for HF Spaces SDK.
"""
import streamlit as st
import pandas as pd
import json
import os
import sys

# ── Path resolution ──────────────────────────────────────────────────────────
# app.py lives at repo root, so src/ and artifacts/ are siblings
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, 'src'))

JD_FEATURES_PATH = os.path.join(_root, 'artifacts', 'jd_features.json')

# ── Imports from src/ ─────────────────────────────────────────────────────────
from scorer import semantic_skill_match
from reasoning_gen import generate_reasoning
from feature_engineer import calculate_career_quality_score, calculate_logistics_score
from behavioral import calculate_behavioral_multiplier
from honeypot import check_honeypot
from veto_checker import check_veto

# ── Embedded sample data ──────────────────────────────────────────────────────
SAMPLE_CANDIDATES = [
    {
        "candidate_id": "CAND_SAMPLE_001",
        "name": "Arjun Mehta",
        "years_of_experience": 7,
        "current_location": "Pune",
        "willing_to_relocate": True,
        "notice_period_days": 30,
        "career_history": [
            {
                "company": "Swiggy", "title": "Senior AI Engineer",
                "start_date": "2021-03-01", "end_date": None, "is_current": True,
                "description": "Built embedding-based semantic retrieval system using FAISS and Elasticsearch. Led ML model serving, Python, vector databases, dense retrieval, semantic search, ranking evaluation, NDCG, A/B testing.",
                "company_size": "1001-5000"
            },
            {
                "company": "Freshworks", "title": "ML Engineer",
                "start_date": "2018-06-01", "end_date": "2021-02-28", "is_current": False,
                "description": "Ranking evaluation framework NDCG MRR MAP. Python scikit-learn pandas numpy. Retrieval-augmented generation experiments.",
                "company_size": "1001-5000"
            }
        ],
        "skills": [
            {"name": "embeddings", "proficiency": "expert", "duration_months": 38, "endorsements": 45},
            {"name": "vector database", "proficiency": "advanced", "duration_months": 26, "endorsements": 30},
            {"name": "python", "proficiency": "expert", "duration_months": 84, "endorsements": 80},
            {"name": "ranking evaluation", "proficiency": "advanced", "duration_months": 22, "endorsements": 20},
            {"name": "qlora", "proficiency": "intermediate", "duration_months": 10, "endorsements": 8}
        ],
        "redrob_signals": {
            "signup_date": "2022-01-15", "last_active_date": "2026-06-20",
            "open_to_work_flag": True, "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.75, "avg_response_time_hours": 8,
            "profile_completeness_score": 92, "connection_count": 450, "endorsements_received": 45,
            "email_verified": True, "phone_verified": True, "linkedin_verified": True,
            "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 35, "max": 50},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "interview_completion_rate": 0.9, "offer_acceptance_rate": 0.8,
            "profile_views_received_30d": 22, "skill_assessment_scores": [85, 90],
            "github_activity_score": 65, "search_appearance_30d": 18, "saved_by_recruiters_30d": 7
        }
    },
    {
        "candidate_id": "CAND_SAMPLE_002",
        "name": "Priya Nair",
        "years_of_experience": 6,
        "current_location": "Hyderabad",
        "willing_to_relocate": True,
        "notice_period_days": 45,
        "career_history": [
            {
                "company": "Razorpay", "title": "AI Research Engineer",
                "start_date": "2022-01-01", "end_date": None, "is_current": True,
                "description": "Semantic search and retrieval-augmented generation (RAG) pipeline. Weaviate Qdrant vector database. Python transformers sentence-transformers. LLM fine-tuning LoRA PEFT.",
                "company_size": "1001-5000"
            },
            {
                "company": "Hotstar", "title": "Machine Learning Engineer",
                "start_date": "2019-07-01", "end_date": "2021-12-31", "is_current": False,
                "description": "Learning-to-rank LTR LambdaMART recommendation systems. Python numpy pandas A/B testing offline evaluation.",
                "company_size": "5001-10000"
            }
        ],
        "skills": [
            {"name": "rag", "proficiency": "expert", "duration_months": 28, "endorsements": 38},
            {"name": "weaviate", "proficiency": "advanced", "duration_months": 20, "endorsements": 18},
            {"name": "python", "proficiency": "expert", "duration_months": 72, "endorsements": 62},
            {"name": "learning-to-rank", "proficiency": "advanced", "duration_months": 24, "endorsements": 25},
            {"name": "lora", "proficiency": "intermediate", "duration_months": 14, "endorsements": 12}
        ],
        "redrob_signals": {
            "signup_date": "2021-08-10", "last_active_date": "2026-06-18",
            "open_to_work_flag": True, "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.82, "avg_response_time_hours": 6,
            "profile_completeness_score": 88, "connection_count": 380, "endorsements_received": 38,
            "email_verified": True, "phone_verified": True, "linkedin_verified": True,
            "notice_period_days": 45, "expected_salary_range_inr_lpa": {"min": 32, "max": 48},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "interview_completion_rate": 0.85, "offer_acceptance_rate": 0.75,
            "profile_views_received_30d": 18, "skill_assessment_scores": [88, 82],
            "github_activity_score": 72, "search_appearance_30d": 14, "saved_by_recruiters_30d": 6
        }
    },
    {
        "candidate_id": "CAND_SAMPLE_003",
        "name": "Karthik Rajan",
        "years_of_experience": 5,
        "current_location": "Mumbai",
        "willing_to_relocate": False,
        "notice_period_days": 60,
        "career_history": [
            {
                "company": "Zepto", "title": "ML Engineer",
                "start_date": "2022-06-01", "end_date": None, "is_current": True,
                "description": "Search relevance and ranking systems. Elasticsearch hybrid search vector index. Python numpy. Offline evaluation NDCG metrics. Embedding models.",
                "company_size": "501-1000"
            },
            {
                "company": "Ola", "title": "Data Scientist",
                "start_date": "2020-09-01", "end_date": "2022-05-31", "is_current": False,
                "description": "Predictive modelling Python scikit-learn feature engineering recommendation.",
                "company_size": "5001-10000"
            }
        ],
        "skills": [
            {"name": "elasticsearch", "proficiency": "advanced", "duration_months": 24, "endorsements": 22},
            {"name": "python", "proficiency": "expert", "duration_months": 60, "endorsements": 45},
            {"name": "embeddings", "proficiency": "intermediate", "duration_months": 18, "endorsements": 14},
            {"name": "ndcg", "proficiency": "intermediate", "duration_months": 20, "endorsements": 10}
        ],
        "redrob_signals": {
            "signup_date": "2023-03-01", "last_active_date": "2026-05-10",
            "open_to_work_flag": True, "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.60, "avg_response_time_hours": 20,
            "profile_completeness_score": 78, "connection_count": 220, "endorsements_received": 22,
            "email_verified": True, "phone_verified": False, "linkedin_verified": True,
            "notice_period_days": 60, "expected_salary_range_inr_lpa": {"min": 25, "max": 38},
            "preferred_work_mode": "hybrid", "willing_to_relocate": False,
            "interview_completion_rate": 0.70, "offer_acceptance_rate": 0.60,
            "profile_views_received_30d": 10, "skill_assessment_scores": [75],
            "github_activity_score": 40, "search_appearance_30d": 8, "saved_by_recruiters_30d": 3
        }
    },
    {
        "candidate_id": "CAND_SAMPLE_004",
        "name": "Rahul Sharma",
        "years_of_experience": 8,
        "current_location": "Delhi NCR",
        "willing_to_relocate": False,
        "notice_period_days": 30,
        "career_history": [
            {
                "company": "TCS", "title": "Senior Consultant",
                "start_date": "2018-01-01", "end_date": None, "is_current": True,
                "description": "Enterprise consulting delivery. Client engagement project management. Python automation.",
                "company_size": "10001+"
            },
            {
                "company": "Infosys", "title": "Technology Analyst",
                "start_date": "2015-06-01", "end_date": "2017-12-31", "is_current": False,
                "description": "Software delivery consulting engagement management.",
                "company_size": "10001+"
            }
        ],
        "skills": [
            {"name": "python", "proficiency": "intermediate", "duration_months": 36, "endorsements": 15},
            {"name": "project management", "proficiency": "advanced", "duration_months": 60, "endorsements": 30},
            {"name": "machine learning", "proficiency": "beginner", "duration_months": 6, "endorsements": 5}
        ],
        "redrob_signals": {
            "signup_date": "2023-09-01", "last_active_date": "2026-06-01",
            "open_to_work_flag": True, "applications_submitted_30d": 8,
            "recruiter_response_rate": 0.50, "avg_response_time_hours": 48,
            "profile_completeness_score": 70, "connection_count": 300, "endorsements_received": 30,
            "email_verified": True, "phone_verified": True, "linkedin_verified": False,
            "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 28, "max": 40},
            "preferred_work_mode": "onsite", "willing_to_relocate": False,
            "interview_completion_rate": 0.50, "offer_acceptance_rate": 0.40,
            "profile_views_received_30d": 6, "skill_assessment_scores": [],
            "github_activity_score": 10, "search_appearance_30d": 4, "saved_by_recruiters_30d": 1
        }
    },
    {
        "candidate_id": "CAND_SAMPLE_005",
        "name": "Sneha Kapoor",
        "years_of_experience": 10,
        "current_location": "Noida",
        "willing_to_relocate": True,
        "notice_period_days": 90,
        "career_history": [
            {
                "company": "Jio", "title": "Senior ML Engineer",
                "start_date": "2020-04-01", "end_date": None, "is_current": True,
                "description": "Dense retrieval and semantic search. RAG pipeline retrieval-augmented generation. Python sentence-transformers FAISS vector database Pinecone. Ranking NDCG A/B testing evaluation framework.",
                "company_size": "10001+"
            },
            {
                "company": "Paytm", "title": "ML Engineer",
                "start_date": "2017-08-01", "end_date": "2020-03-31", "is_current": False,
                "description": "Recommendation and ranking systems. LambdaMART learning-to-rank Python numpy pandas offline evaluation.",
                "company_size": "5001-10000"
            },
            {
                "company": "Walmart Labs", "title": "Software Engineer",
                "start_date": "2015-07-01", "end_date": "2017-07-31", "is_current": False,
                "description": "Backend Python services. Search and retrieval systems.",
                "company_size": "10001+"
            }
        ],
        "skills": [
            {"name": "dense retrieval", "proficiency": "expert", "duration_months": 42, "endorsements": 55},
            {"name": "faiss", "proficiency": "expert", "duration_months": 36, "endorsements": 40},
            {"name": "python", "proficiency": "expert", "duration_months": 108, "endorsements": 90},
            {"name": "ranking evaluation", "proficiency": "advanced", "duration_months": 32, "endorsements": 28},
            {"name": "lambdamart", "proficiency": "advanced", "duration_months": 24, "endorsements": 22},
            {"name": "sentence-transformers", "proficiency": "expert", "duration_months": 38, "endorsements": 35}
        ],
        "redrob_signals": {
            "signup_date": "2021-05-20", "last_active_date": "2026-06-22",
            "open_to_work_flag": True, "applications_submitted_30d": 4,
            "recruiter_response_rate": 0.70, "avg_response_time_hours": 12,
            "profile_completeness_score": 95, "connection_count": 600, "endorsements_received": 55,
            "email_verified": True, "phone_verified": True, "linkedin_verified": True,
            "notice_period_days": 90, "expected_salary_range_inr_lpa": {"min": 45, "max": 65},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "interview_completion_rate": 0.88, "offer_acceptance_rate": 0.70,
            "profile_views_received_30d": 28, "skill_assessment_scores": [92, 88, 85],
            "github_activity_score": 78, "search_appearance_30d": 22, "saved_by_recruiters_30d": 9
        }
    },
    {
        "candidate_id": "CAND_SAMPLE_006",
        "name": "Deepak HONEYPOT",
        "years_of_experience": 15,
        "current_location": "Pune",
        "willing_to_relocate": True,
        "notice_period_days": 0,
        "career_history": [
            {
                "company": "StartupX", "title": "AI Engineer",
                "start_date": "2024-01-01", "end_date": None, "is_current": True,
                "description": "Some AI work.",
                "company_size": "1-10"
            }
        ],
        "skills": [
            {"name": "embeddings", "proficiency": "expert", "duration_months": 0, "endorsements": 200},
            {"name": "vector database", "proficiency": "expert", "duration_months": 0, "endorsements": 180},
            {"name": "python", "proficiency": "expert", "duration_months": 0, "endorsements": 220},
            {"name": "transformer", "proficiency": "expert", "duration_months": 0, "endorsements": 190},
            {"name": "kubernetes", "proficiency": "expert", "duration_months": 0, "endorsements": 175},
            {"name": "ray", "proficiency": "expert", "duration_months": 0, "endorsements": 160},
            {"name": "lora", "proficiency": "expert", "duration_months": 0, "endorsements": 155},
            {"name": "qlora", "proficiency": "expert", "duration_months": 0, "endorsements": 145},
            {"name": "faiss", "proficiency": "expert", "duration_months": 0, "endorsements": 140},
            {"name": "weaviate", "proficiency": "expert", "duration_months": 0, "endorsements": 135},
            {"name": "pinecone", "proficiency": "expert", "duration_months": 0, "endorsements": 130}
        ],
        "redrob_signals": {
            "signup_date": "2024-01-10", "last_active_date": "2025-11-01",
            "open_to_work_flag": True, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.02, "avg_response_time_hours": 200,
            "profile_completeness_score": 40, "connection_count": 10, "endorsements_received": 5,
            "email_verified": False, "phone_verified": False, "linkedin_verified": False,
            "notice_period_days": 0, "expected_salary_range_inr_lpa": {"min": 0, "max": 200},
            "preferred_work_mode": "remote", "willing_to_relocate": True,
            "interview_completion_rate": 0.0, "offer_acceptance_rate": 0.0,
            "profile_views_received_30d": 0, "skill_assessment_scores": [],
            "github_activity_score": 0, "search_appearance_30d": 0, "saved_by_recruiters_30d": 0
        }
    }
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legend Acers — Candidate Ranker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Legend Acers")
    st.markdown("**India.Runs 2026** — Track 01: Data & AI")
    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
- **S1** JD Understanding
- **S2a** Honeypot Detection (7 traps)
- **S2b** JD VETO Checker (6 rules)
- **S3** Feature Engineering
- **S4** Semantic Skill Match (MiniLM-L6-v2)
- **S5** Behavioral Multiplier (23 signals)
- **S6** Per-Candidate Reasoning
    """)
    st.divider()
    st.markdown("### Scoring Formula")
    st.code("Raw = 0.45·Career + 0.40·Semantic + 0.15·Logistics\nFinal = Raw × Behavioral[0.3–1.0]", language=None)
    st.divider()
    st.markdown("### Team Legend Acers")
    st.markdown("Akhilesh · Kishore · Sanggit · Venkatram")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Intelligent Candidate Discovery")
st.markdown(
    "*Rank candidates the way a great recruiter would — not by matching keywords, "
    "but by actually understanding who fits the role.*"
)
st.divider()

# ── Load JD Features ──────────────────────────────────────────────────────────
@st.cache_data
def load_jd_features():
    try:
        with open(JD_FEATURES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

jd_features = load_jd_features()
if jd_features is None:
    st.error("`artifacts/jd_features.json` not found. Ensure the file exists in the repository.")
    st.stop()

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown("### Upload Candidates or Try Sample Data")

col_up, col_samp = st.columns([2, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload a JSON file (array of candidate profiles, ≤ 100 candidates)",
        type=["json"],
        help="Each candidate must follow the Redrob candidate schema with career_history, skills, and redrob_signals fields."
    )
with col_samp:
    st.markdown("&nbsp;")
    use_sample = st.button(
        "🎲 Load 6 Sample Candidates",
        help="Loads 6 built-in synthetic candidates: 3 strong matches, 1 consulting VETO, 1 borderline, 1 honeypot",
        use_container_width=True
    )

# ── Resolve candidate list ────────────────────────────────────────────────────
candidates = None

if use_sample:
    candidates = SAMPLE_CANDIDATES
    st.success(
        f"Loaded **{len(candidates)} sample candidates** — includes strong matches, a consulting VETO, "
        "and a honeypot profile to demonstrate all detection layers."
    )

elif uploaded_file is not None:
    try:
        candidates = json.load(uploaded_file)
        if isinstance(candidates, dict):
            candidates = [candidates]
    except json.JSONDecodeError:
        st.error("Invalid JSON. Please upload a valid JSON array of candidate objects.")
        st.stop()

    if len(candidates) > 100:
        st.warning(f"Uploaded {len(candidates)} candidates — only the first 100 will be processed.")
        candidates = candidates[:100]
    st.success(f"Loaded **{len(candidates)} candidates** from uploaded file.")

# ── Run pipeline ──────────────────────────────────────────────────────────────
if candidates is not None:
    col_run, _ = st.columns([1, 3])
    with col_run:
        run_clicked = st.button("🚀 Run Ranker", type="primary", use_container_width=True)

    if run_clicked:
        yoe_min = jd_features.get("experience_range", {}).get("min", 5)
        yoe_max = jd_features.get("experience_range", {}).get("max", 9)
        target_loc = jd_features.get("target_locations", [])

        progress_bar = st.progress(0, text="Initialising pipeline...")
        results = []
        honeypot_count = 0
        veto_count = 0

        for idx, cand in enumerate(candidates):
            cid = cand.get("candidate_id", f"UNKNOWN_{idx}")

            is_honeypot = check_honeypot(cand)
            is_veto, veto_reason = check_veto(cand, jd_features)
            combined_veto = is_honeypot or is_veto

            if is_honeypot:
                honeypot_count += 1
            if is_veto:
                veto_count += 1

            career   = calculate_career_quality_score(cand.get("career_history", []), yoe_min, yoe_max)
            logistics = calculate_logistics_score(cand, target_loc)
            behavioral = calculate_behavioral_multiplier(cand.get("redrob_signals", {}))
            semantic  = semantic_skill_match(cand.get("skills", []), jd_features)

            raw   = 0.45 * career + 0.40 * semantic + 0.15 * logistics
            score = raw * behavioral if not combined_veto else 0.0

            results.append({
                "candidate_id": cid,
                "score": score,
                "veto": combined_veto,
                "veto_reason": veto_reason if is_veto else ("honeypot" if is_honeypot else None),
                "semantic": semantic,
                "career": career,
                "logistics": logistics,
                "behavioral": behavioral,
                "candidate_obj": cand
            })

            progress_bar.progress(
                (idx + 1) / len(candidates),
                text=f"Processing {idx + 1} / {len(candidates)} — {cid}"
            )

        progress_bar.empty()

        # Sort descending score, tiebreak by candidate_id
        results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        top_results = results[:min(100, len(results))]

        output_data = []
        for i, res in enumerate(top_results):
            rank = i + 1
            reasoning = generate_reasoning(
                res["candidate_obj"], jd_features,
                res["semantic"], res["behavioral"],
                rank, res["veto"],
                veto_reason=res.get("veto_reason")
            )
            output_data.append({
                "candidate_id": res["candidate_id"],
                "rank": rank,
                "score": round(res["score"], 4),
                "reasoning": reasoning,
                "_semantic":   round(res["semantic"],   4),
                "_career":     round(res["career"],     4),
                "_logistics":  round(res["logistics"],  4),
                "_behavioral": round(res["behavioral"], 4),
                "_flagged":    res["veto"]
            })

        df = pd.DataFrame(output_data)

        # ── Metrics ──────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Results Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Processed", len(candidates))
        c2.metric("Honeypots Caught", honeypot_count)
        c3.metric("JD VETOs", veto_count)
        c4.metric("Top Score", f"{df['score'].max():.4f}")
        c5.metric("Score Range", f"{df['score'].min():.4f} – {df['score'].max():.4f}")

        # ── Ranked table ──────────────────────────────────────────────────────
        st.markdown("### Ranked Candidates")
        st.dataframe(
            df[["rank", "candidate_id", "score", "reasoning"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank":         st.column_config.NumberColumn("Rank", width="small"),
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
                "score":        st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "reasoning":    st.column_config.TextColumn("Reasoning", width="large"),
            }
        )

        # ── Score breakdown ───────────────────────────────────────────────────
        with st.expander("📊 Score Breakdown per Candidate", expanded=False):
            st.dataframe(
                df[["rank", "candidate_id", "score", "_semantic", "_career", "_logistics", "_behavioral", "_flagged"]].rename(columns={
                    "_semantic": "Semantic", "_career": "Career",
                    "_logistics": "Logistics", "_behavioral": "Behavioral×", "_flagged": "Flagged?"
                }),
                use_container_width=True,
                hide_index=True
            )

        # ── Bias comparison panel ─────────────────────────────────────────────
        st.markdown("### ⚖️ Traditional ATS vs. Legend Acers")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("#### ❌ Traditional Keyword ATS")
            st.markdown("""
- Ranks by **keyword count** + years of experience
- Ignores behavioral availability signals
- **Cannot detect fabricated profiles** (honeypots pass through)
- Misses semantic equivalences: "RAG" ≈ "retrieval-augmented generation"
- Penalises career changers with transferable skills
            """)
        with cb:
            st.markdown("#### ✅ Legend Acers System")
            st.markdown(f"""
- **MiniLM-L6-v2 semantic matching** — cosine similarity, not keyword count
- **All 23 Redrob behavioral signals** as multiplier (recency decay, response rate, work mode…)
- Detected **{honeypot_count} honeypot(s)** via 7 trap rules → hard 0.0 score
- Applied **{veto_count} JD VETO(s)** (consulting-only, architect no-code, pure research…)
- Weights: Career (45%) + Semantic (40%) + Logistics (15%) × Behavioral
- **Per-candidate reasoning** referencing real profile data — no templates
            """)

        # ── Download ──────────────────────────────────────────────────────────
        st.divider()
        csv_bytes = df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download submission.csv",
            csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True
        )
