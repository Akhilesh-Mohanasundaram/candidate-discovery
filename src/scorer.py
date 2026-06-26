"""
Scorer — Stage 4: Semantic Skill Matching + Score Fusion
==========================================================
Calculates semantic skill similarity between candidate skills and JD requirements
using all-MiniLM-L6-v2 embeddings with cosine similarity.

Score Fusion:
  Raw Score = 0.45 × Career Quality + 0.40 × Semantic Skill Match + 0.15 × Logistics
  Final Score = Raw Score × Behavioral Multiplier [0.3 – 1.0]

VETO/honeypot candidates always return 0.0.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError):
    SentenceTransformer = None
    print("Warning: SentenceTransformer unavailable. Will use mock embeddings.")

_model = None
_cached_target_skills = None      # The list of skill strings last encoded
_cached_target_embeddings = None   # The corresponding embeddings matrix


def get_model():
    global _model
    if _model is None:
        if SentenceTransformer is not None:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            class MockModel:
                def encode(self, texts):
                    # Return deterministic random vectors of dim 384
                    np.random.seed(42)
                    return np.random.rand(len(texts), 384)
            _model = MockModel()
    return _model


def _get_target_embeddings(jd_features):
    """
    Returns (target_skills_list, target_embeddings_matrix).
    Caches across calls so we don't re-encode the same JD skills 100K times.
    """
    global _cached_target_skills, _cached_target_embeddings

    # Build the target skills list from JD features
    target_skills = []
    for req in jd_features.get("hard_requirements", []):
        target_skills.append(req["skill"])
        target_skills.extend(req.get("aliases", []))
    for pref in jd_features.get("soft_preferences", []):
        target_skills.append(pref["skill"])
        target_skills.extend(pref.get("aliases", []))

    # Check if we already have these cached
    if (_cached_target_skills is not None and
            _cached_target_skills == target_skills and
            _cached_target_embeddings is not None):
        return target_skills, _cached_target_embeddings

    # Encode and cache
    model = get_model()
    _cached_target_skills = target_skills
    _cached_target_embeddings = model.encode(target_skills)
    return target_skills, _cached_target_embeddings


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

    target_skills, target_embeddings = _get_target_embeddings(jd_features)

    if not target_skills:
        return 0.5  # Fallback if JD parsing failed

    model = get_model()

    proficiency_weights = {
        "beginner": 0.3,
        "intermediate": 0.6,
        "advanced": 0.8,
        "expert": 1.0
    }

    hard_reqs = jd_features.get("hard_requirements", [])
    soft_prefs = jd_features.get("soft_preferences", [])
    max_possible_score = len(hard_reqs) * 1.0

    target_to_req = []
    req_idx = 0
    for req in hard_reqs:
        target_to_req.append(req_idx)
        for _ in req.get("aliases", []):
            target_to_req.append(req_idx)
        req_idx += 1
    for pref in soft_prefs:
        target_to_req.append(req_idx)
        for _ in pref.get("aliases", []):
            target_to_req.append(req_idx)
        req_idx += 1

    best_score_per_req = [0.0] * req_idx

    for c_skill in candidate_skills:
        name = c_skill.get("name", "")
        prof = c_skill.get("proficiency", "beginner")
        endorsements = c_skill.get("endorsements", 0)
        duration = c_skill.get("duration_months", 0)

        if precomputed_skill_embeddings and name in precomputed_skill_embeddings:
            c_emb = precomputed_skill_embeddings[name]
        else:
            c_emb = model.encode([name])[0]

        similarities = cosine_similarity([c_emb], target_embeddings)[0]

        prof_w = proficiency_weights.get(prof, 0.5)
        dur_w = 0.0 if duration == 0 else min(1.0, duration / 36.0)
        end_w = 1.0 + min(0.2, endorsements / 100.0)
        c_weight = prof_w * dur_w * end_w

        for t_idx, sim in enumerate(similarities):
            if sim > 0.4:
                score = sim * c_weight
                req_i = target_to_req[t_idx]
                if score > best_score_per_req[req_i]:
                    best_score_per_req[req_i] = score

    total_score = sum(best_score_per_req)

    # Normalize score between 0 and 1
    normalized = min(1.0, total_score / max(1.0, max_possible_score))
    return normalized


def calculate_final_score(semantic_score, career_quality_score, logistics_score,
                          behavioral_multiplier, is_veto=False):
    """
    Final score computation.
    Raw Score = 0.45 * Career + 0.40 * Semantic + 0.15 * Logistics
    Final Score = Raw Score * Behavioral Multiplier [0.3 - 1.0]

    VETO candidates (honeypots, disqualifiers) return 0.0 immediately.
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
    except Exception:
        jd_feats = {"hard_requirements": [{"skill": "python"}], "soft_preferences": []}

    test_skills = [
        {"name": "Python", "proficiency": "expert", "duration_months": 48, "endorsements": 50},
        {"name": "Sentence Transformers", "proficiency": "advanced", "duration_months": 12, "endorsements": 10}
    ]

    score = semantic_skill_match(test_skills, jd_feats)
    print(f"Test Semantic Score: {score}")

    final = calculate_final_score(score, 0.8, 0.9, 0.85)
    print(f"Test Final Score: {final}")
