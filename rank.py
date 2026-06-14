import os
import sys
import json
import numpy as np
import time
import gzip
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from scorer import calculate_final_score
from reasoning_gen import generate_reasoning

def run_ranker():
    start_time = time.time()
    
    # 1. Load feature matrix
    try:
        matrix = np.load('artifacts/feature_matrix.npz')
    except FileNotFoundError:
        print("Error: artifacts/feature_matrix.npz not found. Run precompute.py first.")
        return
        
    c_ids = matrix['candidate_ids']
    career = matrix['career_scores']
    semantic = matrix['semantic_scores']
    logistics = matrix['logistics_scores']
    behavioral = matrix['behavioral_mults']
    vetoes = matrix['is_veto_flags']
    
    # 2. Vectorized scoring
    # Calculate raw score: 0.45 Career, 0.40 Semantic, 0.15 Logistics
    raw_scores = 0.45 * career + 0.40 * semantic + 0.15 * logistics
    final_scores = raw_scores * behavioral
    
    # Apply vetoes (set to 0.0)
    final_scores[vetoes] = 0.0
    
    # 3. Sort and get top 100
    # We need to sort by final_scores DESC, and then by c_ids ASC (to break ties).
    # Round scores to 4 decimal places to remove floating point noise that prevents exact ties.
    rounded_scores = np.round(final_scores, 4)
    sort_indices = np.lexsort((c_ids, -rounded_scores))
    top_indices = sort_indices[:100]
    
    top_cids = c_ids[top_indices]
    top_scores = final_scores[top_indices]
    top_veto = vetoes[top_indices]
    top_behavioral = behavioral[top_indices]
    top_semantic = semantic[top_indices]
    
    top_cid_set = set(top_cids)
    
    # 4. Fetch raw candidate data for reasoning generation
    data_dir = r'dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge'
    jsonl_path = os.path.join(data_dir, 'candidates.jsonl')
    gz_path = os.path.join(data_dir, 'candidates.jsonl.gz')
    sample_path = os.path.join(data_dir, 'sample_candidates.json')
    
    candidate_objects = {}
    
    # We only scan the file to find the 100 objects we need
    if os.path.exists(jsonl_path):
        f = open(jsonl_path, 'r', encoding='utf-8')
        iterator = f
    elif os.path.exists(gz_path):
        f = gzip.open(gz_path, 'rt', encoding='utf-8')
        iterator = f
    else:
        f = open(sample_path, 'r', encoding='utf-8')
        iterator = json.load(f)
        
    for line in iterator:
        if isinstance(line, str):
            if not line.strip(): continue
            cand = json.loads(line)
        else:
            cand = line
            
        cid = cand.get("candidate_id")
        if cid in top_cid_set:
            candidate_objects[cid] = cand
            if len(candidate_objects) == 100:
                break # Found all we need
                
    if hasattr(f, 'close'):
        f.close()
        
    # Load JD features for reasoning
    with open('artifacts/jd_features.json', 'r', encoding='utf-8') as f:
        jd_features = json.load(f)
        
    # 5. Generate reasoning and write CSV
    out_csv = 'submission.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        
        for i, idx in enumerate(top_indices):
            cid = c_ids[idx]
            score = final_scores[idx]
            veto = vetoes[idx]
            beh = behavioral[idx]
            sem = semantic[idx]
            
            cand_obj = candidate_objects.get(cid, {})
            
            # Since rank 1..100
            rank = i + 1
            
            # Generate reasoning
            reasoning = generate_reasoning(
                candidate=cand_obj,
                jd_features=jd_features,
                semantic_score=sem,
                behavioral_multiplier=beh,
                final_rank=rank,
                is_veto=veto
            )
            
            writer.writerow([cid, rank, round(score, 4), reasoning])
            
    elapsed = time.time() - start_time
    print(f"Ranking complete in {elapsed:.2f} seconds.")
    print(f"Output saved to {out_csv}")

if __name__ == "__main__":
    run_ranker()
