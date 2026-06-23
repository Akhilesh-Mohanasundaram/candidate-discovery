"""
rank.py — Main Ranking Script
==============================
Produces submission.csv containing the Top 100 candidate ranking.
Uses pre-computed feature matrix (from precompute.py) for sub-5-minute execution.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Compute constraints: ≤5 min wall-clock, ≤16 GB RAM, CPU only, no network calls.
"""
import os
import sys
import json
import numpy as np
import time
import gzip
import csv
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from scorer import calculate_final_score
from reasoning_gen import generate_reasoning


def run_ranker(candidates_path=None, output_path='submission.csv'):
    start_time = time.time()

    # 1. Load feature matrix
    matrix_path = 'artifacts/feature_matrix.npz'
    try:
        matrix = np.load(matrix_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: {matrix_path} not found. Run precompute.py first.")
        return

    c_ids = matrix['candidate_ids']
    career = matrix['career_scores']
    semantic = matrix['semantic_scores']
    logistics = matrix['logistics_scores']
    behavioral = matrix['behavioral_mults']
    vetoes = matrix['is_veto_flags']

    # Load veto reasons if available (new field from updated precompute)
    veto_reasons_arr = None
    if 'veto_reasons' in matrix:
        veto_reasons_arr = matrix['veto_reasons']

    print(f"Loaded feature matrix: {len(c_ids)} candidates")
    print(f"  Vetoed candidates: {np.sum(vetoes.astype(bool))}")

    # 2. Vectorized scoring
    # Weights: 45% Career Quality, 40% Semantic Skill Match, 15% Logistics
    # Behavioral applied as a multiplier [0.3 - 1.0]
    raw_scores = 0.45 * career + 0.40 * semantic + 0.15 * logistics
    final_scores = raw_scores * behavioral

    # Apply vetoes (honeypot/disqualified → score = 0.0)
    final_scores[vetoes.astype(bool)] = 0.0

    # 3. Sort and get top 100
    # Sort by final_scores DESC, then by candidate_id ASC (deterministic tiebreak)
    rounded_scores = np.round(final_scores, 6)
    sort_indices = np.lexsort((c_ids, -rounded_scores))
    top_indices = sort_indices[:100]

    top_cids = c_ids[top_indices]
    top_scores = final_scores[top_indices]
    top_veto = vetoes[top_indices]
    top_behavioral = behavioral[top_indices]
    top_semantic = semantic[top_indices]

    top_cid_set = set(top_cids.tolist())

    # 4. Fetch raw candidate data for reasoning generation
    index_path = 'artifacts/candidate_index.json'
    candidate_objects = {}
    if os.path.exists(index_path):
        print(f"Loading candidate index from {index_path}...")
        with open(index_path, 'r', encoding='utf-8') as f:
            candidate_index = json.load(f)
            for cid in top_cids:
                if cid in candidate_index:
                    candidate_objects[cid] = candidate_index[cid]
    else:
        print("Warning: candidate_index.json not found. Reasoning will be minimal.")

    print(f"Retrieved {len(candidate_objects)} candidate profiles for reasoning.")

    # Load JD features for reasoning
    jd_path = 'artifacts/jd_features.json'
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_features = json.load(f)

    # 5. Generate reasoning and write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])

        for i, idx in enumerate(top_indices):
            cid = str(c_ids[idx])
            score = float(final_scores[idx])
            veto = bool(vetoes[idx])
            beh = float(behavioral[idx])
            sem = float(semantic[idx])

            # Get veto reason if available
            veto_reason = None
            if veto and veto_reasons_arr is not None:
                veto_reason = str(veto_reasons_arr[idx])
                if veto_reason == "":
                    veto_reason = None

            cand_obj = candidate_objects.get(cid, {})
            rank = i + 1

            reasoning = generate_reasoning(
                candidate=cand_obj,
                jd_features=jd_features,
                semantic_score=sem,
                behavioral_multiplier=beh,
                final_rank=rank,
                is_veto=veto,
                veto_reason=veto_reason
            )

            writer.writerow([cid, rank, round(score, 4), reasoning])

    elapsed = time.time() - start_time
    print(f"\nRanking complete in {elapsed:.2f} seconds.")
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rank candidates and produce submission CSV.',
        epilog='Example: python rank.py --candidates ./candidates.jsonl --out ./submission.csv'
    )
    parser.add_argument('--candidates', default=None,
                        help='Path to candidates.jsonl, .jsonl.gz, or .json file')
    parser.add_argument('--out', default='submission.csv',
                        help='Output CSV path (default: submission.csv)')
    args = parser.parse_args()
    run_ranker(candidates_path=args.candidates, output_path=args.out)
