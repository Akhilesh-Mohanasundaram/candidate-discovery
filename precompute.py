"""
precompute.py — Offline Feature Extraction
============================================
Scans all candidates and pre-computes features into artifacts/feature_matrix.npz.
This step may exceed 5 minutes — that is allowed by the spec.
The ranking step (rank.py) must complete within 5 minutes.

Usage:
    python precompute.py --candidates ./candidates.jsonl
"""
import os
import sys
import json
import numpy as np
import gzip
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from jd_parser import parse_jd
from honeypot import check_honeypot
from veto_checker import check_veto
from feature_engineer import calculate_career_quality_score, calculate_logistics_score
from behavioral import calculate_behavioral_multiplier
from scorer import semantic_skill_match


def precompute(candidates_path=None):
    print("Starting precomputation...")

    # 1. Load JD Features
    jd_features_path = 'artifacts/jd_features.json'
    if not os.path.exists(jd_features_path):
        print("JD features not found. Generating...")
        jd_features = parse_jd("MOCK JD")
        os.makedirs('artifacts', exist_ok=True)
        with open(jd_features_path, 'w', encoding='utf-8') as f:
            json.dump(jd_features, f, indent=2)
    else:
        with open(jd_features_path, 'r', encoding='utf-8') as f:
            jd_features = json.load(f)

    expected_yoe_min = jd_features.get("experience_range", {}).get("min", 5)
    expected_yoe_max = jd_features.get("experience_range", {}).get("max", 9)
    target_locations = jd_features.get("target_locations", [])

    # 2. Resolve candidate data source
    if candidates_path:
        source_path = candidates_path
    else:
        # Fallback: try common locations (dataset/ is the hackathon bundle)
        for p in [
            os.path.join('dataset', 'candidates.jsonl'),
            os.path.join('dataset', 'candidates.jsonl.gz'),
            os.path.join('dataset', 'sample_candidates.json'),
            'candidates.jsonl',
            'candidates.jsonl.gz',
        ]:
            if os.path.exists(p):
                source_path = p
                break
        else:
            print("Error: No candidate data file found. Provide --candidates path.")
            return

    print(f"Using: {source_path}")

    candidate_ids = []
    career_scores = []
    semantic_scores = []
    logistics_scores = []
    behavioral_mults = []
    is_honeypot_flags = []
    is_veto_flags = []
    veto_reasons = []
    
    candidate_index = {}

    # Open file based on extension
    if source_path.endswith('.gz'):
        f = gzip.open(source_path, 'rt', encoding='utf-8')
        is_jsonl = True
    elif source_path.endswith('.json'):
        with open(source_path, 'r', encoding='utf-8') as jf:
            candidates_list = json.load(jf)
        f = None
        is_jsonl = False
    else:
        f = open(source_path, 'r', encoding='utf-8')
        is_jsonl = True

    honeypot_count = 0
    veto_count = 0

    def process_candidate(cand):
        nonlocal honeypot_count, veto_count

        c_id = cand.get("candidate_id")

        # Stage 2: Honeypot detection (fabricated profiles)
        honeypot = check_honeypot(cand)
        if honeypot:
            honeypot_count += 1

        # JD VETO disqualifiers (misaligned but real profiles)
        veto, veto_reason = check_veto(cand, jd_features)
        if veto:
            veto_count += 1

        # Combined veto flag: either honeypot OR JD disqualifier
        combined_veto = honeypot or veto

        # Stage 3: Feature engineering
        career_q = calculate_career_quality_score(
            cand.get("career_history", []), expected_yoe_min, expected_yoe_max
        )
        logistics = calculate_logistics_score(cand, target_locations)

        # Stage 5: Behavioral multiplier (all 23 signals)
        behavioral = calculate_behavioral_multiplier(cand.get("redrob_signals", {}))

        # Stage 4: Semantic skill match
        sem = semantic_skill_match(cand.get("skills", []), jd_features)

        candidate_ids.append(c_id)
        career_scores.append(career_q)
        semantic_scores.append(sem)
        logistics_scores.append(logistics)
        behavioral_mults.append(behavioral)
        is_honeypot_flags.append(honeypot)
        is_veto_flags.append(combined_veto)
        veto_reasons.append(veto_reason if veto_reason else ("honeypot" if honeypot else ""))

        candidate_index[c_id] = {
            "profile": {
                "years_of_experience": cand.get("profile", {}).get("years_of_experience", 0),
                "current_title": cand.get("profile", {}).get("current_title", "Engineer"),
                "location": cand.get("profile", {}).get("location", "undisclosed location"),
            },
            "skills": [{"name": s.get("name"), "proficiency": s.get("proficiency"), "duration_months": s.get("duration_months")} for s in cand.get("skills", [])],
            "career_history": [{"title": j.get("title"), "company": j.get("company"), "duration_months": j.get("duration_months")} for j in cand.get("career_history", [])],
            "redrob_signals": {
                "recruiter_response_rate": cand.get("redrob_signals", {}).get("recruiter_response_rate", 1.0),
                "notice_period_days": cand.get("redrob_signals", {}).get("notice_period_days", 0),
                "last_active_date": cand.get("redrob_signals", {}).get("last_active_date", ""),
                "open_to_work_flag": cand.get("redrob_signals", {}).get("open_to_work_flag", False),
                "github_activity_score": cand.get("redrob_signals", {}).get("github_activity_score", -1),
                "interview_completion_rate": cand.get("redrob_signals", {}).get("interview_completion_rate", 1.0),
                "willing_to_relocate": cand.get("redrob_signals", {}).get("willing_to_relocate", False)
            }
        }

    count = 0
    if is_jsonl and f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            process_candidate(cand)
            count += 1
            if count % 10000 == 0:
                print(f"  Processed {count} candidates... "
                      f"(honeypots: {honeypot_count}, vetoes: {veto_count})")
        f.close()
    elif not is_jsonl:
        for cand in candidates_list:
            process_candidate(cand)
            count += 1
            if count % 10000 == 0:
                print(f"  Processed {count} candidates... "
                      f"(honeypots: {honeypot_count}, vetoes: {veto_count})")

    # 3. Save feature matrix and candidate index
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/candidate_index.json', 'w', encoding='utf-8') as f:
        json.dump(candidate_index, f)
        
    np.savez_compressed(
        'artifacts/feature_matrix.npz',
        candidate_ids=np.array(candidate_ids),
        career_scores=np.array(career_scores, dtype=np.float32),
        semantic_scores=np.array(semantic_scores, dtype=np.float32),
        logistics_scores=np.array(logistics_scores, dtype=np.float32),
        behavioral_mults=np.array(behavioral_mults, dtype=np.float32),
        is_honeypot_flags=np.array(is_honeypot_flags, dtype=bool),
        is_veto_flags=np.array(is_veto_flags, dtype=bool),
        veto_reasons=np.array(veto_reasons)
    )
    print(f"\nPrecomputation complete. Processed {count} candidates.")
    print(f"  Honeypots detected: {honeypot_count}")
    print(f"  JD VETO disqualified: {veto_count}")
    print(f"  Total vetoed: {honeypot_count + veto_count}")
    print("Saved to artifacts/feature_matrix.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pre-compute feature matrix from candidate data.',
        epilog='Example: python precompute.py --candidates ./candidates.jsonl'
    )
    parser.add_argument('--candidates', default=None,
                        help='Path to candidates.jsonl, .jsonl.gz, or .json file')
    args = parser.parse_args()
    precompute(candidates_path=args.candidates)
