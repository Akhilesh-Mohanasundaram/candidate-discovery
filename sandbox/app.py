"""
Legend Acers — Intelligent Candidate Discovery Sandbox
=======================================================
Local development version of the Streamlit demo app.
Run with: cd sandbox && streamlit run app.py

For HuggingFace Spaces deployment, see app.py at the repo root.
"""
import streamlit as st
import pandas as pd
import json
import os
import sys

# --- Path resolution for both local and HF Spaces deployment ---
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_this_dir, '..'))
_src_dir = os.path.join(_repo_root, 'src')

# Add src to path (works whether app is run from sandbox/ or repo root)
if os.path.isdir(_src_dir):
    sys.path.insert(0, _src_dir)
else:
    # Fallback: if src is in same dir (flat HF Space deployment)
    sys.path.insert(0, _this_dir)

from scorer import semantic_skill_match, calculate_final_score
from reasoning_gen import generate_reasoning
from feature_engineer import calculate_career_quality_score, calculate_logistics_score
from behavioral import calculate_behavioral_multiplier
from honeypot import check_honeypot
from veto_checker import check_veto

# --- JD Features path resolution ---
_artifacts_dir = os.path.join(_repo_root, 'artifacts')
if not os.path.isdir(_artifacts_dir):
    _artifacts_dir = os.path.join(_this_dir, 'artifacts')
JD_FEATURES_PATH = os.path.join(_artifacts_dir, 'jd_features.json')

# --- Page config ---
st.set_page_config(
    page_title="Legend Acers - Candidate Ranker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚡ Legend Acers")
    st.markdown("**India.Runs 2026** | Track 01")
    st.divider()
    st.markdown("### How It Works")
    st.markdown("""
    1. Upload a JSON file with ≤100 candidates
    2. Click **Run Ranker**
    3. View ranked results with reasoning
    4. Download the CSV submission
    """)
    st.divider()
    st.markdown("### Pipeline Stages")
    st.markdown("""
    - **S1** JD Understanding
    - **S2** Honeypot Detection (7 traps)
    - **S3** Feature Engineering
    - **S4** Semantic Skill Match (MiniLM)
    - **S5** Behavioral Multiplier (23 signals)
    - **S6** Reasoning Generation
    """)
    st.divider()
    st.markdown("### Scoring Formula")
    st.code("Final = (0.45*Career + 0.40*Semantic + 0.15*Logistics) * Behavioral", language=None)

# --- Main content ---
st.title("⚡ Intelligent Candidate Discovery")
st.markdown("*Rank candidates the way a great recruiter would — not by matching keywords, "
            "but by actually understanding who fits the role.*")

# --- Load JD Features ---
@st.cache_data
def load_jd_features():
    try:
        with open(JD_FEATURES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

jd_features = load_jd_features()
if jd_features is None:
    st.error("JD Features file not found. Ensure `artifacts/jd_features.json` exists.")
    st.stop()

# --- File upload ---
st.markdown("### Upload Candidates")
uploaded_file = st.file_uploader(
    "Upload a JSON file with candidate profiles (≤100 candidates)",
    type=["json"],
    help="The file should contain an array of candidate objects matching the candidates.jsonl schema."
)

if uploaded_file is not None:
    try:
        candidates = json.load(uploaded_file)
        if isinstance(candidates, dict):
            # Handle single candidate wrapped in object
            candidates = [candidates]
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON array of candidates.")
        st.stop()

    if len(candidates) > 100:
        st.warning(f"Uploaded {len(candidates)} candidates. Only the first 100 will be processed.")
        candidates = candidates[:100]

    st.success(f"Loaded **{len(candidates)}** candidates.")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button("🚀 Run Ranker", type="primary", use_container_width=True)

    if run_clicked:
        yoe_min = jd_features.get("experience_range", {}).get("min", 5)
        yoe_max = jd_features.get("experience_range", {}).get("max", 9)
        target_loc = jd_features.get("target_locations", [])

        progress_bar = st.progress(0, text="Processing candidates...")
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

            career = calculate_career_quality_score(
                cand.get("career_history", []), yoe_min, yoe_max
            )
            logistics = calculate_logistics_score(cand, target_loc)
            behavioral = calculate_behavioral_multiplier(cand.get("redrob_signals", {}))
            semantic = semantic_skill_match(cand.get("skills", []), jd_features)

            raw = 0.45 * career + 0.40 * semantic + 0.15 * logistics
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

            progress_bar.progress((idx + 1) / len(candidates),
                                  text=f"Processing {idx + 1}/{len(candidates)}...")

        progress_bar.empty()

        # Sort by score descending, tiebreak by candidate_id ascending
        results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        top_results = results[:min(100, len(results))]

        # Generate output
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
                "_semantic": round(res["semantic"], 4),
                "_career": round(res["career"], 4),
                "_logistics": round(res["logistics"], 4),
                "_behavioral": round(res["behavioral"], 4),
                "_honeypot": res["veto"]
            })

        df = pd.DataFrame(output_data)

        # --- Metrics ---
        st.markdown("---")
        st.markdown("### Results Overview")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Candidates Processed", len(candidates))
        m2.metric("Honeypots Detected", honeypot_count)
        m3.metric("JD Vetoes", veto_count)
        m4.metric("Top Score", f"{df['score'].max():.4f}")
        m5.metric("Score Range", f"{df['score'].min():.4f} - {df['score'].max():.4f}")

        # --- Ranked Table ---
        st.markdown("### Ranked Candidates")
        display_cols = ["rank", "candidate_id", "score", "reasoning"]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
                "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            }
        )

        # --- Score Breakdown (expandable) ---
        with st.expander("📊 Detailed Score Breakdown", expanded=False):
            detail_cols = ["rank", "candidate_id", "score", "_semantic", "_career",
                           "_logistics", "_behavioral", "_honeypot"]
            st.dataframe(
                df[detail_cols].rename(columns={
                    "_semantic": "Semantic",
                    "_career": "Career",
                    "_logistics": "Logistics",
                    "_behavioral": "Behavioral",
                    "_honeypot": "Honeypot?"
                }),
                use_container_width=True,
                hide_index=True
            )

        # --- Bias Comparison Panel ---
        st.markdown("### ⚖️ Bias Comparison: Traditional ATS vs. Our System")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### ❌ Traditional ATS")
            st.markdown("""
            - Ranks by **keyword count** and **years of experience**
            - Ignores behavioral availability signals
            - Cannot detect fabricated profiles (honeypots)
            - Misses semantic skill equivalences (e.g., "RAG" ≈ "retrieval-augmented generation")
            - Penalizes career changers with transferable skills
            """)

        with col_b:
            st.markdown("#### ✅ Legend Acers System")
            st.markdown(f"""
            - Uses **MiniLM-L6-v2 semantic matching** (cosine similarity)
            - Applies **all 23 Redrob behavioral signals** as multiplier (recency decay, response rate, salary, work mode, etc.)
            - Detects **{honeypot_count} honeypot(s)** via 7 trap rules (scored 0.0)
            - Applies **{veto_count} JD VETO disqualifier(s)** (pure research, consulting-only, etc.)
            - Weights: Career (45%) + Semantic (40%) + Logistics (15%) × Behavioral
            - Generates candidate-specific reasoning referencing real profile data
            """)

        # --- Download ---
        st.markdown("---")
        csv_cols = ["candidate_id", "rank", "score", "reasoning"]
        csv_data = df[csv_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download submission.csv",
            csv_data, "submission.csv", "text/csv",
            use_container_width=True
        )
