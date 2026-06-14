import os
import sys
import json
import numpy as np
import gzip

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from jd_parser import parse_jd
from honeypot import check_honeypot
from feature_engineer import calculate_career_quality_score, calculate_logistics_score
from behavioral import calculate_behavioral_multiplier
from scorer import semantic_skill_match

def precompute():
    print("Starting precomputation...")
    
    # 1. Load JD Features
    jd_features_path = 'artifacts/jd_features.json'
    if not os.path.exists(jd_features_path):
        print("JD features not found. Generating...")
        jd_features = parse_jd("MOCK JD")
    else:
        with open(jd_features_path, 'r', encoding='utf-8') as f:
            jd_features = json.load(f)
            
    expected_yoe_min = jd_features.get("experience_range", {}).get("min", 5)
    expected_yoe_max = jd_features.get("experience_range", {}).get("max", 9)
    target_locations = jd_features.get("target_locations", [])
    
    # 2. Iterate candidates
    # We will use sample_candidates.json for local testing, 
    # but the real system uses candidates.jsonl or candidates.jsonl.gz
    
    data_dir = r'dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge'
    gz_path = os.path.join(data_dir, 'candidates.jsonl.gz')
    jsonl_path = os.path.join(data_dir, 'candidates.jsonl')
    sample_path = os.path.join(data_dir, 'sample_candidates.json')
    
    candidate_ids = []
    career_scores = []
    semantic_scores = []
    logistics_scores = []
    behavioral_mults = []
    is_veto_flags = []
    
    # Fallback logic for which file to use
    if os.path.exists(jsonl_path):
        print(f"Using {jsonl_path}")
        f = open(jsonl_path, 'r', encoding='utf-8')
        iterator = f
    elif os.path.exists(gz_path):
        print(f"Using {gz_path}")
        f = gzip.open(gz_path, 'rt', encoding='utf-8')
        iterator = f
    else:
        print(f"Using {sample_path}")
        f = open(sample_path, 'r', encoding='utf-8')
        candidates_list = json.load(f)
        iterator = candidates_list
        
    count = 0
    for line in iterator:
        if isinstance(line, str):
            if not line.strip():
                continue
            cand = json.loads(line)
        else:
            cand = line # If it's a dict from sample_candidates
            
        c_id = cand.get("candidate_id")
        
        # Honeypot Check
        veto = check_honeypot(cand)
        
        # Features
        career_q = calculate_career_quality_score(cand.get("career_history", []), expected_yoe_min, expected_yoe_max)
        logistics = calculate_logistics_score(cand, target_locations)
        behavioral = calculate_behavioral_multiplier(cand.get("redrob_signals", {}))
        
        # Semantic Score
        sem = semantic_skill_match(cand.get("skills", []), jd_features)
        
        candidate_ids.append(c_id)
        career_scores.append(career_q)
        semantic_scores.append(sem)
        logistics_scores.append(logistics)
        behavioral_mults.append(behavioral)
        is_veto_flags.append(veto)
        
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} candidates...")
            
    if hasattr(f, 'close'):
        f.close()
        
    # 3. Save feature matrix
    os.makedirs('artifacts', exist_ok=True)
    np.savez_compressed(
        'artifacts/feature_matrix.npz',
        candidate_ids=np.array(candidate_ids),
        career_scores=np.array(career_scores),
        semantic_scores=np.array(semantic_scores),
        logistics_scores=np.array(logistics_scores),
        behavioral_mults=np.array(behavioral_mults),
        is_veto_flags=np.array(is_veto_flags)
    )
    print(f"Precomputation complete. Processed {count} candidates.")
    print("Saved to artifacts/feature_matrix.npz")

if __name__ == "__main__":
    precompute()
