import json
import os
import sys

# Add src to python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scorer import semantic_skill_match, calculate_final_score
from reasoning_gen import generate_reasoning

def test_run():
    # Load JD features
    jd_features_path = 'artifacts/jd_features.json'
    with open(jd_features_path, 'r', encoding='utf-8') as f:
        jd_features = json.load(f)
        
    # Load sample candidates
    dataset_path = os.path.join('dataset', 'sample_candidates.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
        
    print(f"Loaded {len(candidates)} candidates from the official dataset.\n")
    print("-" * 60)
    
    # Process the first 3 candidates
    for i, candidate in enumerate(candidates[:3]):
        cid = candidate.get("candidate_id")
        name = candidate.get("profile", {}).get("anonymized_name", "Unknown")
        title = candidate.get("profile", {}).get("current_title", "Unknown")
        
        print(f"Candidate {i+1}: {cid} ({name} - {title})")
        
        # 1. Semantic Skill Match
        skills = candidate.get("skills", [])
        semantic_score = semantic_skill_match(skills, jd_features)
        
        # 2. Mocking Member B's Feature & Behavioral components for the test
        career_score = 0.8  # Mock career quality score
        logistics_score = 0.9 # Mock logistics score
        behavioral_multiplier = 0.85 # Mock behavioral multiplier
        
        # 3. Final Scoring
        final_score = calculate_final_score(
            semantic_score, 
            career_score, 
            logistics_score, 
            behavioral_multiplier, 
            is_veto=False
        )
        
        # 4. Reasoning Generation
        reasoning = generate_reasoning(
            candidate, 
            jd_features, 
            semantic_score, 
            behavioral_multiplier, 
            final_rank=i+1, 
            is_veto=False
        )
        
        print(f"  Semantic Match Score: {semantic_score:.3f}")
        print(f"  Final Projected Score: {final_score:.3f}")
        print(f"  Generated Reasoning: {reasoning}")
        print("-" * 60)

if __name__ == "__main__":
    test_run()
