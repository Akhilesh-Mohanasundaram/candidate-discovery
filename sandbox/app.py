import streamlit as st
import pandas as pd
import json
import os
import sys

# Add parent to path to import ranker components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from scorer import semantic_skill_match, calculate_final_score
from reasoning_gen import generate_reasoning
from feature_engineer import calculate_career_quality_score, calculate_logistics_score
from behavioral import calculate_behavioral_multiplier
from honeypot import check_honeypot

st.set_page_config(page_title="Legend Acers Ranker", page_icon="⚡", layout="wide")

st.title("⚡ Legend Acers Candidate Ranker")
st.markdown("Intelligent Candidate Discovery - Streamlit Sandbox")

uploaded_file = st.file_uploader("Upload candidates (JSON)", type=["json"])

if uploaded_file is not None:
    candidates = json.load(uploaded_file)
    st.success(f"Loaded {len(candidates)} candidates.")
    
    if st.button("Run Ranker"):
        with st.spinner("Processing..."):
            # Load JD
            jd_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'jd_features.json')
            try:
                with open(jd_path, 'r') as f:
                    jd_features = json.load(f)
            except:
                st.error("JD Features not found. Please run precompute locally.")
                st.stop()
                
            yoe_min = jd_features.get("experience_range", {}).get("min", 5)
            yoe_max = jd_features.get("experience_range", {}).get("max", 9)
            target_loc = jd_features.get("target_locations", [])
            
            results = []
            for cand in candidates:
                cid = cand.get("candidate_id")
                veto = check_honeypot(cand)
                
                career = calculate_career_quality_score(cand.get("career_history", []), yoe_min, yoe_max)
                logistics = calculate_logistics_score(cand, target_loc)
                behavioral = calculate_behavioral_multiplier(cand.get("redrob_signals", {}))
                semantic = semantic_skill_match(cand.get("skills", []), jd_features)
                
                raw = 0.45 * career + 0.40 * semantic + 0.15 * logistics
                score = raw * behavioral if not veto else 0.0
                
                results.append({
                    "candidate_id": cid,
                    "score": score,
                    "veto": veto,
                    "semantic": semantic,
                    "behavioral": behavioral,
                    "candidate_obj": cand
                })
                
            # Sort
            results.sort(key=lambda x: x["score"], reverse=True)
            top_100 = results[:100]
            
            output_data = []
            for i, res in enumerate(top_100):
                rank = i + 1
                reasoning = generate_reasoning(
                    res["candidate_obj"], jd_features, res["semantic"], res["behavioral"], rank, res["veto"]
                )
                output_data.append({
                    "candidate_id": res["candidate_id"],
                    "rank": rank,
                    "score": round(res["score"], 4),
                    "reasoning": reasoning
                })
                
            df = pd.DataFrame(output_data)
            st.write("### Top Ranked Candidates")
            st.dataframe(df)
            
            # Bias Comparison Panel
            st.write("### Bias Comparison: Traditional vs Legend Acers")
            st.markdown("""
            **Traditional ATS** would rank strictly by Keyword Count and Years of Experience.
            **Our Ranker** penalizes 'Fake Experts' (0.0 score) and boosts high-behavioral-engagement candidates.
            """)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download submission.csv", csv, "submission.csv", "text/csv")
