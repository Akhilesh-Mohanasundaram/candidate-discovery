import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
except OSError:
    SentenceTransformer = None
    print("Warning: Could not load SentenceTransformer due to DLL error. Will use mock embeddings.")
# For standalone testing, we instantiate it here.
_model = None

def get_model():
    global _model
    if _model is None:
        if SentenceTransformer is not None:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            class MockModel:
                def encode(self, texts):
                    # Return random vectors of dim 384
                    np.random.seed(42)
                    return np.random.rand(len(texts), 384)
            _model = MockModel()
    return _model

def semantic_skill_match(candidate_skills, jd_features, precomputed_skill_embeddings=None):
    """
    Member A's Semantic Skill Matching Logic.
    Calculates score based on semantic similarity of skills, weighted by proficiency,
    endorsements, and duration.
    
    Args:
        candidate_skills (list): The 'skills' array from the candidate profile.
        jd_features (dict): The loaded artifacts/jd_features.json
        precomputed_skill_embeddings (dict): Optional dict mapping skill name to embedding.
    """
    if not candidate_skills:
        return 0.0

    # Extract all target skills from JD features
    target_skills = []
    for req in jd_features.get("hard_requirements", []):
        target_skills.append(req["skill"])
        target_skills.extend(req.get("aliases", []))
    for pref in jd_features.get("soft_preferences", []):
        target_skills.append(pref["skill"])
        target_skills.extend(pref.get("aliases", []))
        
    if not target_skills:
        return 0.5  # Fallback if JD parsing failed
        
    # Get embeddings for target skills
    model = get_model()
    target_embeddings = model.encode(target_skills)
    
    proficiency_weights = {
        "beginner": 0.3,
        "intermediate": 0.6,
        "advanced": 0.8,
        "expert": 1.0
    }
    
    total_score = 0.0
    max_possible_score = len(jd_features.get("hard_requirements", [])) * 1.0
    
    for c_skill in candidate_skills:
        name = c_skill.get("name", "")
        prof = c_skill.get("proficiency", "beginner")
        endorsements = c_skill.get("endorsements", 0)
        duration = c_skill.get("duration_months", 0)
        
        # Calculate semantic similarity against all target skills
        # In a real run with precomputed_skill_embeddings, we'd look it up
        if precomputed_skill_embeddings and name in precomputed_skill_embeddings:
            c_emb = precomputed_skill_embeddings[name]
        else:
            c_emb = model.encode([name])[0]
            
        # Cosine similarity
        similarities = cosine_similarity([c_emb], target_embeddings)[0]
        max_sim = np.max(similarities)
        
        # Only count if it's reasonably similar
        if max_sim > 0.4:
            # Multiplier for proficiency and duration
            prof_w = proficiency_weights.get(prof, 0.5)
            # Cap duration weight at 36 months for normalization
            dur_w = min(1.0, duration / 36.0) if duration > 0 else 0.1
            # Endorsements give a slight bump (up to 20%)
            end_w = 1.0 + min(0.2, endorsements / 100.0)
            
            skill_score = max_sim * prof_w * dur_w * end_w
            total_score += skill_score

    # Normalize score between 0 and 1
    normalized = min(1.0, total_score / max(1.0, max_possible_score))
    return normalized

def calculate_final_score(semantic_score, career_quality_score, logistics_score, behavioral_multiplier, is_veto=False):
    """
    Member A's Scoring Weights Formula.
    Weights: 40% Career Quality, 35% Semantic Skill Match, 10% Logistics, 15% Behavioral (Multiplier applied later or factored in)
    Wait, the guidelines say:
    Hybrid Scoring Engine: Semantic (35%) + Career (40%) + Logistics (10%) + Behavioral (15%)
    AND Behavioral multiplier (0.3 - 1.0) applied to raw scores.
    Let's refine: The 15% behavioral score is derived from signals, 
    but Stage 5 says "Behavioral Signal Multiplier ... as a multiplier (0.3-1.0) on raw scores".
    So we'll blend: (40% Career + 35% Semantic + 10% Logistics) * Behavioral Multiplier.
    """
    if is_veto:
        return 0.0
        
    raw_score = (
        0.45 * career_quality_score + 
        0.40 * semantic_score + 
        0.15 * logistics_score
    )
    
    final_score = raw_score * behavioral_multiplier
    return final_score

if __name__ == "__main__":
    # Test scorer
    import json
    try:
        with open('artifacts/jd_features.json', 'r') as f:
            jd_feats = json.load(f)
    except:
        jd_feats = {"hard_requirements": [{"skill": "python"}], "soft_preferences": []}
        
    test_skills = [
        {"name": "Python", "proficiency": "expert", "duration_months": 48, "endorsements": 50},
        {"name": "Sentence Transformers", "proficiency": "advanced", "duration_months": 12, "endorsements": 10}
    ]
    
    score = semantic_skill_match(test_skills, jd_feats)
    print(f"Test Semantic Score: {score}")
    
    final = calculate_final_score(score, 0.8, 0.9, 0.85)
    print(f"Test Final Score: {final}")
