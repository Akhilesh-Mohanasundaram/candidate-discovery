import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from scorer import semantic_skill_match

@pytest.fixture
def jd_features():
    return {"hard_requirements": [{"skill": "python", "aliases": []}], "soft_preferences": []}

def test_scorer_saturation(jd_features):
    cand_skills = [
        {"name": "Python", "proficiency": "expert", "duration_months": 48},
        {"name": "Advanced Python", "proficiency": "expert", "duration_months": 48}
    ]
    score = semantic_skill_match(cand_skills, jd_features)
    assert score <= 1.0

def test_scorer_zero_duration(jd_features):
    cand_skills = [{"name": "Python", "proficiency": "expert", "duration_months": 0}]
    score = semantic_skill_match(cand_skills, jd_features)
    assert score == 0.0

def test_scorer_endorsements(jd_features):
    cand1 = [{"name": "Python", "proficiency": "expert", "duration_months": 36, "endorsements": 0}]
    cand2 = [{"name": "Python", "proficiency": "expert", "duration_months": 36, "endorsements": 50}]
    score1 = semantic_skill_match(cand1, jd_features)
    score2 = semantic_skill_match(cand2, jd_features)
    assert score2 > score1

def test_scorer_threshold(jd_features):
    cand_skills = [{"name": "Baking Cookies", "proficiency": "expert", "duration_months": 60}]
    score = semantic_skill_match(cand_skills, jd_features)
    assert score == 0.0
