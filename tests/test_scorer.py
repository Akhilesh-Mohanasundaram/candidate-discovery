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
