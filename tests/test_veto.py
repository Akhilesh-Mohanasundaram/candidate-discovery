import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from veto_checker import check_veto

@pytest.fixture
def jd_features():
    return {"hard_requirements": [], "soft_preferences": [], "experience_range": {"min": 5, "max": 9}, "target_locations": []}

def test_veto_consulting(jd_features):
    cand = {
        "profile": {"years_of_experience": 8, "current_title": "Consultant"},
        "career_history": [{"title": "Consultant", "company": "TCS", "duration_months": 96, "start_date": "2018-01-01"}]
    }
    veto, reason = check_veto(cand, jd_features)
    assert veto
    assert reason == "consulting_only"

def test_veto_architect(jd_features):
    cand = {
        "profile": {"years_of_experience": 10, "current_title": "Chief Technology Officer"},
        "career_history": [{"title": "Chief Technology Officer", "company": "Startup", "duration_months": 24, "start_date": "2024-01-01"}]
    }
    veto, reason = check_veto(cand, jd_features)
    assert veto
    assert reason == "architect_no_code"
